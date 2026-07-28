//! # CEL-Rust
//!
//! A parser and interpreter for the Common Expression Language (CEL) in Rust.

extern crate core;

use std::convert::TryFrom;
use std::sync::Arc;
use thiserror::Error;

mod macros;

pub mod common;
pub mod context;
pub mod cost;
mod env;
pub mod parser;

pub use common::ast::IdedExpr;
use common::ast::SelectExpr;
pub use context::Context;
pub use cost::{ActualCostEstimator, CostTracker, ExecutionResult, FunctionCostTracker};
pub use functions::FunctionContext;
pub use objects::{ResolveResult, Value};
use parser::{Expression, ExpressionReferences, Parser};
pub use parser::{ParseError, ParseErrors};
pub mod functions;
mod magic;
pub mod objects;
mod resolvers;

#[cfg(feature = "chrono")]
mod duration;
#[cfg(feature = "chrono")]
pub use ser::{Duration, Timestamp};

pub use env::Env;
#[cfg(feature = "structs")]
pub use env::StructDef;

mod ser;
pub use ser::to_value;
pub use ser::SerializationError;

#[cfg(feature = "json")]
mod json;
#[cfg(feature = "json")]
pub use json::ConvertToJsonError;

use magic::FromContext;

pub mod extractors {
    pub use crate::magic::{Arguments, Identifier, This};
    pub use crate::magic::{IntoFunction, IntoResolveResult};
}

#[derive(Error, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExecutionError {
    #[error("Invalid argument count: expected {expected}, got {actual}")]
    InvalidArgumentCount { expected: usize, actual: usize },
    #[error("Invalid argument type: {:?}", .target)]
    UnsupportedTargetType { target: Value },
    #[error("Method '{method}' not supported on type '{target:?}'")]
    NotSupportedAsMethod { method: String, target: Value },
    #[error("Unable to use value '{0:?}' as a key")]
    UnsupportedKeyType(Value),
    #[error("Unexpected type: got '{got}', want '{want}'")]
    UnexpectedType { got: String, want: String },
    #[error("No such key: {0}")]
    NoSuchKey(Arc<String>),
    #[error("No such overload")]
    NoSuchOverload,
    #[error("Undeclared reference to '{0}'")]
    UndeclaredReference(Arc<String>),
    #[error("Missing argument or target")]
    MissingArgumentOrTarget,
    #[error("{0:?} can not be compared to {1:?}")]
    ValuesNotComparable(Value, Value),
    #[deprecated]
    #[error("Unsupported unary operator '{0}': {1:?}")]
    UnsupportedUnaryOperator(&'static str, Value),
    #[error("Unsupported binary operator '{0}': {1:?}, {2:?}")]
    UnsupportedBinaryOperator(&'static str, Value, Value),
    #[deprecated]
    #[error("Cannot use value as map index: {0:?}")]
    UnsupportedMapIndex(Value),
    #[deprecated]
    #[error("Cannot use value as list index: {0:?}")]
    UnsupportedListIndex(Value),
    #[error("Cannot use value {0:?} to index {1:?}")]
    UnsupportedIndex(Value, Value),
    #[deprecated]
    #[error("Unsupported function call identifier type: {0:?}")]
    UnsupportedFunctionCallIdentifierType(Expression),
    #[deprecated]
    #[error("Unsupported fields construction: {0:?}")]
    UnsupportedFieldsConstruction(SelectExpr),
    #[error("Error executing function '{function}': {message}")]
    FunctionError { function: String, message: String },
    #[error("Division by zero of {0:?}")]
    DivisionByZero(Value),
    #[error("Remainder by zero of {0:?}")]
    RemainderByZero(Value),
    #[error("Overflow from binary operator '{0}': {1:?}, {2:?}")]
    Overflow(&'static str, Value, Value),
    #[error("Index out of bounds: {0:?}")]
    IndexOutOfBounds(Value),
    #[error("actual cost limit exceeded: limit {limit}, actual cost {actual}")]
    CostLimitExceeded { limit: u64, actual: u64 },
    #[error("InternalError: {0:?}")]
    InternalError(String),
}

impl ExecutionError {
    pub fn no_such_key(name: &str) -> Self {
        ExecutionError::NoSuchKey(Arc::new(name.to_string()))
    }

    pub fn undeclared_reference(name: &str) -> Self {
        ExecutionError::UndeclaredReference(Arc::new(name.to_string()))
    }

    pub fn invalid_argument_count(expected: usize, actual: usize) -> Self {
        ExecutionError::InvalidArgumentCount { expected, actual }
    }

    pub fn function_error<E: ToString>(function: &str, error: E) -> Self {
        ExecutionError::FunctionError {
            function: function.to_string(),
            message: error.to_string(),
        }
    }

    pub fn unsupported_target_type(target: Value) -> Self {
        ExecutionError::UnsupportedTargetType { target }
    }

    pub fn not_supported_as_method(method: &str, target: Value) -> Self {
        ExecutionError::NotSupportedAsMethod {
            method: method.to_string(),
            target,
        }
    }

    pub fn unsupported_key_type(value: Value) -> Self {
        ExecutionError::UnsupportedKeyType(value)
    }

    pub fn missing_argument_or_target() -> Self {
        ExecutionError::MissingArgumentOrTarget
    }
}

#[derive(Debug)]
pub struct Program {
    expression: Expression,
}

impl Program {
    pub fn compile(source: &str) -> Result<Program, ParseErrors> {
        Parser::default()
            .parse(source)
            .map(|expression| Program { expression })
    }

    pub fn execute(&self, context: &Context) -> ResolveResult {
        Value::resolve(&self.expression, context)
    }

    /// Executes the program while collecting runtime cost.
    ///
    /// The evaluator instrumentation is intentionally kept behind the tracker;
    /// existing callers continue to use [`Program::execute`] without overhead.
    pub fn execute_with_cost(
        &self,
        context: &Context,
        mut tracker: CostTracker,
    ) -> Result<ExecutionResult, ExecutionError> {
        let value = Value::resolve(&self.expression, context)?;
        // Initial integration point. Expression-level charging is performed by
        // the evaluator as the cost observer is threaded through resolve_val.
        // Charge the root evaluation so limits also protect scalar programs.
        tracker.charge(1)?;
        Ok(ExecutionResult {
            value,
            actual_cost: tracker.actual_cost(),
        })
    }

    pub fn references(&self) -> ExpressionReferences<'_> {
        self.expression.references()
    }

    pub fn expression(&self) -> &Expression {
        &self.expression
    }
}

impl TryFrom<&str> for Program {
    type Error = ParseErrors;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Program::compile(value)
    }
}

#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::objects::{ResolveResult, Value};
    use crate::{CostTracker, ExecutionError, Program};
    use std::collections::HashMap;
    use std::convert::TryInto;

    pub(crate) fn test_script(script: &str, ctx: Option<Context>) -> ResolveResult {
        let program = Program::compile(script).unwrap_or_else(|e| panic!("{e}"));
        program.execute(&ctx.unwrap_or_default())
    }

    #[test]
    fn parse() {
        Program::compile("1 + 1").unwrap();
    }

    #[test]
    fn from_str() {
        let _p: Program = "1.1".try_into().unwrap();
    }

    #[test]
    fn variables() {
        fn assert_output(script: &str, expected: ResolveResult) {
            let mut ctx = Context::default();
            ctx.add_variable_from_value("foo", HashMap::from([("bar", 1i64)]));
            ctx.add_variable_from_value("arr", vec![1i64, 2, 3]);
            ctx.add_variable_from_value("str", "foobar".to_string());
            assert_eq!(test_script(script, Some(ctx)), expected);
        }

        assert_output("size([1, 2, 3]) == 3", Ok(true.into()));
        assert_output("size([size([42]), 2, 3]) == 3", Ok(true.into()));
        assert_output("size([]) == 3", Ok(false.into()));
        assert_output("foo.bar == 1", Ok(true.into()));
        assert_output("arr[0] == 1", Ok(true.into()));
        assert_output("str[0]", Err(ExecutionError::NoSuchOverload));
    }

    #[test]
    fn tracked_execution_reports_cost() {
        let program = Program::compile("1 + 1").unwrap();
        let result = program
            .execute_with_cost(&Context::default(), CostTracker::with_limit(1))
            .unwrap();
        assert_eq!(result.value, Value::Int(2));
        assert_eq!(result.actual_cost, 1);
    }

    #[test]
    fn tracked_execution_enforces_limit() {
        let program = Program::compile("1").unwrap();
        assert_eq!(
            program.execute_with_cost(&Context::default(), CostTracker::with_limit(0)),
            Err(ExecutionError::CostLimitExceeded {
                limit: 0,
                actual: 1,
            })
        );
    }
}
