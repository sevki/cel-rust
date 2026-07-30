//! Continuation-passing-style (CPS) evaluation of [`Program`]s.
//!
//! [`Program::execute`] evaluates a single expression and hands the caller
//! back a [`ResolveResult`]. Some use cases want to *chain* evaluations
//! instead: run one program, decide what to do next based on its result
//! (run another program? bind the result and continue? stop?), and so on,
//! without building up nested closures or `match` arms on the call stack.
//!
//! This module models that as a small state machine: a [`Step`] is either a
//! request to run a [`Program`] and pass its result to a continuation, or a
//! final result. [`run`] drives a chain of `Step`s to completion with a
//! plain loop (a *trampoline*): each continuation returns the *next* `Step`
//! instead of calling back into `run` (or into itself) directly, so a chain
//! of any length executes in constant native stack depth, unlike a chain of
//! ordinary nested callbacks (`k1(r1, || k2(r2, || k3(...)))`), where each
//! link adds a stack frame.
//!
//! [`Pipeline`] is a convenience builder for the common case of running a
//! fixed sequence of programs, binding each result to a variable for the
//! next one, and stopping at the first error.
//!
//! # Example
//!
//! ```
//! use cel::cps::{run, Step};
//! use cel::{Context, Program};
//!
//! let double = Program::compile("x * 2").unwrap();
//! let mut ctx = Context::default();
//! ctx.add_variable_from_value("x", 5i64);
//!
//! // Run `double`, bind its result back to `x`, then run it again.
//! let step = Step::cont(&double, |result, ctx| {
//!     let value = result.unwrap();
//!     ctx.add_variable_from_value("x", value);
//!     Step::cont(&double, |result, _ctx| Step::done(result))
//! });
//!
//! assert_eq!(run(step, &mut ctx), Ok(20i64.into()));
//! ```
use crate::{Context, Program, ResolveResult, Value};

/// What to do after a [`Program`] has produced a [`ResolveResult`].
///
/// Build a `Step` with [`Step::cont`] to run a program and react to its
/// result, or [`Step::done`] to stop the chain with a final result. Drive a
/// `Step` to completion with [`run`].
pub enum Step<'p> {
    /// Run `program` against the context passed to [`run`], then pass its
    /// result (and the context, for binding variables) to `k` to obtain the
    /// next `Step`.
    Continue {
        program: &'p Program,
        k: Box<dyn FnOnce(ResolveResult, &mut Context<'p>) -> Step<'p> + 'p>,
    },
    /// Stop the chain, producing a final result.
    Done(ResolveResult),
}

impl<'p> Step<'p> {
    /// Ends the chain with `result` as the final outcome.
    pub fn done(result: ResolveResult) -> Self {
        Step::Done(result)
    }

    /// Runs `program`, then calls `k` with its result (and the shared
    /// context) to determine the next `Step`.
    pub fn cont(
        program: &'p Program,
        k: impl FnOnce(ResolveResult, &mut Context<'p>) -> Step<'p> + 'p,
    ) -> Self {
        Step::Continue {
            program,
            k: Box::new(k),
        }
    }
}

/// Drives a [`Step`] chain to completion against `ctx`.
///
/// This is the trampoline: it repeatedly executes the current step's
/// program and asks its continuation for the next step, looping until a
/// [`Step::Done`] is produced. Because each continuation returns to this
/// loop instead of calling the next step itself, the native call stack
/// depth used by `run` does not grow with the length of the chain.
pub fn run<'p>(mut step: Step<'p>, ctx: &mut Context<'p>) -> ResolveResult {
    loop {
        match step {
            Step::Done(result) => return result,
            Step::Continue { program, k } => {
                let result = program.execute(ctx);
                step = k(result, ctx);
            }
        }
    }
}

/// A fixed sequence of `(variable name, program)` pairs, run in order.
///
/// Each program's result is bound to its associated variable name in the
/// context before the next program runs, so later programs can refer to
/// earlier results by name. Execution stops at the first error.
///
/// # Example
///
/// ```
/// use cel::cps::Pipeline;
/// use cel::{Context, Program};
///
/// let step1 = Program::compile("2 + 2").unwrap();
/// let step2 = Program::compile("total * 10").unwrap();
///
/// let pipeline = Pipeline::new()
///     .then("total", &step1)
///     .then("result", &step2);
///
/// let mut ctx = Context::default();
/// assert_eq!(pipeline.run(&mut ctx), Ok(40i64.into()));
/// ```
pub struct Pipeline<'p> {
    steps: Vec<(&'p str, &'p Program)>,
}

impl<'p> Default for Pipeline<'p> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'p> Pipeline<'p> {
    /// Creates an empty pipeline.
    pub fn new() -> Self {
        Pipeline { steps: Vec::new() }
    }

    /// Appends `program` to the pipeline. Its result will be bound to
    /// `variable` in the context once it runs, making it visible to every
    /// subsequent step.
    pub fn then(mut self, variable: &'p str, program: &'p Program) -> Self {
        self.steps.push((variable, program));
        self
    }

    /// Runs every program in order against `ctx`, stopping at (and
    /// returning) the first error. Returns `Ok(Value::Null)` if the
    /// pipeline has no steps.
    ///
    /// Each intermediate result is bound into `ctx` under its step's
    /// variable name before the next program runs.
    pub fn run(&self, ctx: &mut Context<'_>) -> ResolveResult {
        let mut last = Ok(Value::Null);
        for &(variable, program) in &self.steps {
            let value = program.execute(ctx)?;
            ctx.add_variable_from_value(variable, value.clone());
            last = Ok(value);
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_step() {
        let program = Program::compile("1 + 1").unwrap();
        let mut ctx = Context::default();

        let step = Step::cont(&program, |result, _ctx| Step::done(result));

        assert_eq!(run(step, &mut ctx), Ok(2i64.into()));
    }

    #[test]
    fn chains_and_binds_intermediate_results() {
        let square = Program::compile("x * x").unwrap();
        let add_one = Program::compile("x + 1").unwrap();
        let mut ctx = Context::default();
        ctx.add_variable_from_value("x", 4i64);

        // square(4) = 16, bind to `x`, then add_one(16) = 17
        let step = Step::cont(&square, |result, ctx| {
            let value = result.unwrap();
            ctx.add_variable_from_value("x", value);
            Step::cont(&add_one, |result, _ctx| Step::done(result))
        });

        assert_eq!(run(step, &mut ctx), Ok(17i64.into()));
    }

    #[test]
    fn short_circuits_on_error() {
        let fails = Program::compile("missing_variable").unwrap();
        let never_runs = Program::compile("1 / 0").unwrap();
        let mut ctx = Context::default();

        let mut second_step_ran = false;
        let step = Step::cont(&fails, |result, _ctx| {
            if result.is_err() {
                return Step::done(result);
            }
            second_step_ran = true;
            Step::cont(&never_runs, |result, _ctx| Step::done(result))
        });

        let result = run(step, &mut ctx);
        assert!(result.is_err());
        assert!(!second_step_ran);
    }

    #[test]
    fn branches_on_previous_result() {
        let condition = Program::compile("x > 10").unwrap();
        let big = Program::compile("'big'").unwrap();
        let small = Program::compile("'small'").unwrap();
        let mut ctx = Context::default();
        ctx.add_variable_from_value("x", 100i64);

        let step = Step::cont(&condition, |result, _ctx| {
            if result == Ok(true.into()) {
                Step::cont(&big, |result, _ctx| Step::done(result))
            } else {
                Step::cont(&small, |result, _ctx| Step::done(result))
            }
        });

        assert_eq!(run(step, &mut ctx), Ok("big".into()));
    }

    #[test]
    fn pipeline_runs_in_order() {
        let step1 = Program::compile("2 + 2").unwrap();
        let step2 = Program::compile("total * 10").unwrap();
        let step3 = Program::compile("result - 1").unwrap();

        let pipeline = Pipeline::new()
            .then("total", &step1)
            .then("result", &step2)
            .then("final", &step3);

        let mut ctx = Context::default();
        assert_eq!(pipeline.run(&mut ctx), Ok(39i64.into()));
    }

    #[test]
    fn pipeline_short_circuits_on_error() {
        let ok_step = Program::compile("1").unwrap();
        let fails = Program::compile("undeclared").unwrap();
        let never_runs = Program::compile("1 / 0").unwrap();

        let pipeline = Pipeline::new()
            .then("a", &ok_step)
            .then("b", &fails)
            .then("c", &never_runs);

        let mut ctx = Context::default();
        assert!(pipeline.run(&mut ctx).is_err());
    }

    #[test]
    fn empty_pipeline_returns_null() {
        let pipeline = Pipeline::new();
        let mut ctx = Context::default();
        assert_eq!(pipeline.run(&mut ctx), Ok(Value::Null));
    }

    /// A chain long enough that a non-trampolined implementation (each
    /// continuation calling the next one directly, growing the native call
    /// stack by one frame per step) would overflow the default thread
    /// stack. `run`'s iterative loop keeps native stack depth constant
    /// regardless of chain length, so this completes without issue.
    #[test]
    fn deep_chain_does_not_grow_native_stack() {
        let increment = Program::compile("prev + 1").unwrap();
        let mut ctx = Context::default();
        ctx.add_variable_from_value("prev", 0i64);

        const STEPS: i64 = 200_000;

        fn build(program: &Program, remaining: i64) -> Step<'_> {
            Step::cont(program, move |result, ctx| {
                let value = result.unwrap();
                ctx.add_variable_from_value("prev", value.clone());
                if remaining <= 1 {
                    Step::done(Ok(value))
                } else {
                    build(program, remaining - 1)
                }
            })
        }

        let result = run(build(&increment, STEPS), &mut ctx);
        assert_eq!(result, Ok(STEPS.into()));
    }
}
