//! Demonstrates `cel::cps`: driving a sequence of CEL evaluations through
//! explicit continuations instead of nested return-value plumbing, using the
//! `pipeline!` and `try_step!` macros to cut the boilerplate down.
use cel::cps::{pipeline, run, Step};
use cel::{try_step, Context, Program};

fn main() {
    pipeline_demo();
    branching_demo();
}

/// The common case: a fixed sequence of programs where each result feeds
/// into the next by name. `pipeline!` builds a `Pipeline` from a literal
/// list of steps, which runs them in order and stops at the first error.
fn pipeline_demo() {
    let subtotal = Program::compile("price * quantity").unwrap();
    let with_tax = Program::compile("subtotal + subtotal * tax_rate").unwrap();

    let mut ctx = Context::default();
    ctx.add_variable_from_value("price", 19.99);
    ctx.add_variable_from_value("quantity", 3i64);
    ctx.add_variable_from_value("tax_rate", 0.0725);

    let pipeline = pipeline![
        "subtotal" => &subtotal,
        "total" => &with_tax,
    ];

    let total = pipeline.run(&mut ctx).unwrap();
    println!("order total: {total:?}");
}

/// The general case: `Step` lets a continuation decide what runs next based
/// on the previous result, e.g. choosing between two programs. `run` drives
/// the chain with a plain loop (a trampoline), so this works the same way
/// whether the chain is three steps or three hundred thousand. `try_step!`
/// unwraps each `ResolveResult`, exiting the continuation early on error —
/// the `Step`-chain equivalent of `?`.
fn branching_demo() {
    let is_eligible = Program::compile("age >= 18 && has_id").unwrap();
    let approved = Program::compile("'approved'").unwrap();
    let denied = Program::compile("'denied: must be 18+ with valid ID'").unwrap();

    let mut ctx = Context::default();
    ctx.add_variable_from_value("age", 21i64);
    ctx.add_variable_from_value("has_id", true);

    let step = Step::cont(&is_eligible, |result, _ctx| {
        let eligible = try_step!(result);
        if eligible == true.into() {
            Step::cont(&approved, |result, _ctx| Step::done(result))
        } else {
            Step::cont(&denied, |result, _ctx| Step::done(result))
        }
    });

    let decision = run(step, &mut ctx).unwrap();
    println!("decision: {decision:?}");
    assert_eq!(decision, "approved".into());
}
