//! Runtime cost accounting for CEL evaluation.
//!
//! The formulas mirror CEL-Go's actual-cost tracker: literals and control-flow
//! nodes are free, selects cost one, constructors have a base cost, and calls
//! are charged according to the amount of data they traverse.

use crate::common::ast::{operators, Expr, IdedExpr};
use crate::common::types::{optional::Optional, Kind};
use crate::common::value::Val;
use crate::ExecutionError;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

const STRING_TRAVERSAL_COST_FACTOR: f64 = 0.1;
const REGEX_STRING_LENGTH_COST_FACTOR: f64 = 0.25;
const LIST_CREATE_BASE_COST: u64 = 10;
const MAP_CREATE_BASE_COST: u64 = 30;
const STRUCT_CREATE_BASE_COST: u64 = 40;

/// Supplies an application-specific cost for a function invocation.
///
/// Returning `None` delegates to CEL's standard cost model. The `args` slice
/// includes a receiver as its first item for member calls.
pub trait ActualCostEstimator: Send + Sync {
    fn call_cost(&self, function: &str, args: &[&dyn Val], result: &dyn Val) -> Option<u64>;
}

/// Computes the cost of one overload invocation.
pub type FunctionTracker = Arc<dyn Fn(&[&dyn Val], &dyn Val) -> Option<u64> + Send + Sync>;

/// Builder for [`CostTracker`].
#[derive(Default)]
pub struct CostTrackerBuilder {
    estimator: Option<Arc<dyn ActualCostEstimator>>,
    limit: Option<u64>,
    presence_test_has_cost: bool,
    overload_trackers: HashMap<String, FunctionTracker>,
}

impl CostTrackerBuilder {
    pub fn new() -> Self {
        Self {
            presence_test_has_cost: true,
            ..Self::default()
        }
    }

    pub fn estimator(mut self, estimator: Arc<dyn ActualCostEstimator>) -> Self {
        self.estimator = Some(estimator);
        self
    }

    pub fn limit(mut self, limit: u64) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn presence_test_has_cost(mut self, has_cost: bool) -> Self {
        self.presence_test_has_cost = has_cost;
        self
    }

    pub fn overload_tracker(
        mut self,
        function: impl Into<String>,
        tracker: FunctionTracker,
    ) -> Self {
        self.overload_trackers.insert(function.into(), tracker);
        self
    }

    pub fn build(self) -> CostTracker {
        CostTracker {
            estimator: self.estimator,
            overload_trackers: self.overload_trackers,
            limit: self.limit,
            presence_test_has_cost: self.presence_test_has_cost,
            cost: 0,
            stack: Vec::new(),
        }
    }
}

/// Accumulates the actual cost of a single evaluation.
pub struct CostTracker {
    estimator: Option<Arc<dyn ActualCostEstimator>>,
    overload_trackers: HashMap<String, FunctionTracker>,
    limit: Option<u64>,
    presence_test_has_cost: bool,
    cost: u64,
    stack: Vec<StackValue>,
}

impl Default for CostTracker {
    fn default() -> Self {
        CostTrackerBuilder::new().build()
    }
}

impl CostTracker {
    pub fn builder() -> CostTrackerBuilder {
        CostTrackerBuilder::new()
    }
    pub fn actual_cost(&self) -> u64 {
        self.cost
    }
    pub fn limit(&self) -> Option<u64> {
        self.limit
    }
    pub fn reset(&mut self) {
        self.cost = 0;
        self.stack.clear();
    }

    fn observe(&mut self, expr: &IdedExpr, value: &dyn Val) -> Result<(), ExecutionError> {
        match &expr.expr {
            Expr::Literal(_) => {}
            Expr::Ident(_) => self.cost += 1,
            Expr::Select(select) => {
                self.drop(&[select.operand.id]);
                if !select.test || self.presence_test_has_cost {
                    self.cost += 1;
                }
            }
            Expr::List(list) => {
                self.drop_args(
                    list.elements
                        .iter()
                        .map(|e| e.id)
                        .collect::<Vec<_>>()
                        .as_slice(),
                );
                self.cost += LIST_CREATE_BASE_COST;
            }
            Expr::Map(map) => {
                let ids = map
                    .entries
                    .iter()
                    .flat_map(|e| match &e.expr {
                        crate::common::ast::EntryExpr::MapEntry(e) => vec![e.key.id, e.value.id],
                        crate::common::ast::EntryExpr::StructField(e) => vec![e.value.id],
                    })
                    .collect::<Vec<_>>();
                self.drop_args(&ids);
                self.cost += MAP_CREATE_BASE_COST;
            }
            Expr::Struct(strct) => {
                let ids = strct
                    .entries
                    .iter()
                    .filter_map(|e| match &e.expr {
                        crate::common::ast::EntryExpr::StructField(e) => Some(e.value.id),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                self.drop_args(&ids);
                self.cost += STRUCT_CREATE_BASE_COST;
            }
            Expr::Comprehension(c) => self.drop(&[c.iter_range.id]),
            Expr::Call(call) => {
                if call.func_name == operators::CONDITIONAL {
                    self.drop(&[call.args[0].id, call.args[1].id, call.args[2].id]);
                } else if matches!(
                    call.func_name.as_str(),
                    operators::LOGICAL_AND | operators::LOGICAL_OR
                ) {
                    self.drop(&call.args.iter().map(|a| a.id).collect::<Vec<_>>());
                } else {
                    let mut ids = call.args.iter().map(|a| a.id).collect::<Vec<_>>();
                    if let Some(target) = &call.target {
                        ids.push(target.id);
                    }
                    let mut taken = self.take_args(&ids);
                    if taken.is_none() && call.target.is_some() {
                        let arg_ids = call.args.iter().map(|a| a.id).collect::<Vec<_>>();
                        taken = self.take_args(&arg_ids);
                    }
                    if let Some(mut args) = taken {
                        // Member calls evaluate explicit arguments before their receiver,
                        // but estimators conventionally receive the receiver first.
                        if call.target.is_some() && args.len() > call.args.len() {
                            let target = args.pop().expect("target ID was present");
                            args.insert(0, target);
                        }
                        self.cost += self.call_cost(&call.func_name, &args, value);
                    }
                }
            }
            Expr::Unspecified => {}
        }
        self.stack.push(StackValue {
            value: value.clone_as_boxed(),
            id: expr.id,
        });
        if self.limit.is_some_and(|limit| self.cost > limit) {
            return Err(ExecutionError::CostLimitExceeded);
        }
        Ok(())
    }

    fn call_cost(&self, function: &str, args: &[Box<dyn Val>], result: &dyn Val) -> u64 {
        let refs = args.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        if let Some(tracker) = self.overload_trackers.get(function) {
            if let Some(cost) = tracker(&refs, result) {
                return cost;
            }
        }
        if let Some(estimator) = &self.estimator {
            if let Some(cost) = estimator.call_cost(function, &refs, result) {
                return cost;
            }
        }
        let size = |index| actual_size(refs[index]);
        match function {
            "startsWith" | "endsWith" | "bytes" | "string" | "quote" | "format" => {
                traversal(size(0))
            }
            operators::IN => size(1),
            operators::EQUALS
            | operators::NOT_EQUALS
            | operators::LESS
            | operators::LESS_EQUALS
            | operators::GREATER
            | operators::GREATER_EQUALS => traversal(size(0).min(size(1))),
            operators::ADD
                if refs.len() == 2
                    && matches!(refs[0].get_type().kind(), Kind::String | Kind::Bytes) =>
            {
                traversal(size(0).saturating_add(size(1)))
            }
            "matches" if refs.len() >= 2 => traversal(size(0).saturating_add(1))
                .saturating_mul(((size(1) as f64) * REGEX_STRING_LENGTH_COST_FACTOR).ceil() as u64),
            "contains" if refs.len() >= 2 => traversal(size(0)).saturating_mul(traversal(size(1))),
            _ => 1,
        }
    }

    fn drop(&mut self, ids: &[u64]) {
        for id in ids {
            if let Some(pos) = self.stack.iter().rposition(|v| v.id == *id) {
                self.stack.truncate(pos);
            }
        }
    }

    fn drop_args(&mut self, ids: &[u64]) {
        let _ = self.take_args(ids);
    }

    fn take_args(&mut self, ids: &[u64]) -> Option<Vec<Box<dyn Val>>> {
        if ids
            .iter()
            .any(|id| !self.stack.iter().any(|value| value.id == *id))
        {
            return None;
        }
        let mut result = Vec::with_capacity(ids.len());
        for id in ids.iter().rev() {
            let pos = self.stack.iter().rposition(|v| v.id == *id)?;
            let mut tail = self.stack.split_off(pos);
            result.push(tail.remove(0).value);
        }
        result.reverse();
        Some(result)
    }
}

struct StackValue {
    value: Box<dyn Val>,
    id: u64,
}

fn traversal(size: u64) -> u64 {
    ((size as f64) * STRING_TRAVERSAL_COST_FACTOR).ceil() as u64
}

fn actual_size(value: &dyn Val) -> u64 {
    if let Some(size) = value.as_sizer() {
        return i64::from(size.size()).max(0) as u64;
    }
    if let Some(optional) = value.downcast_ref::<Optional>() {
        if let Some(value) = optional.inner() {
            return actual_size(value);
        }
    }
    1
}

thread_local! { static ACTIVE_TRACKERS: RefCell<Vec<*mut CostTracker>> = const { RefCell::new(Vec::new()) }; }

pub(crate) fn with_tracker<T>(tracker: &mut CostTracker, f: impl FnOnce() -> T) -> T {
    ACTIVE_TRACKERS.with(|trackers| trackers.borrow_mut().push(tracker));
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            ACTIVE_TRACKERS.with(|t| {
                t.borrow_mut().pop();
            });
        }
    }
    let _guard = Guard;
    f()
}

pub(crate) fn observe(expr: &IdedExpr, value: &dyn Val) -> Result<(), ExecutionError> {
    ACTIVE_TRACKERS.with(|trackers| {
        let ptr = trackers.borrow().last().copied();
        // The pointer is installed only for the dynamic extent of `with_tracker`.
        ptr.map_or(Ok(()), |ptr| unsafe { (&mut *ptr).observe(expr, value) })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, Program};

    #[test]
    fn tracks_runtime_branch_and_string_cost() {
        let program =
            Program::compile("false ? 'unused'.contains('x') : 'abcdefghij'.startsWith('a')")
                .unwrap();
        let mut tracker = CostTracker::default();
        program
            .execute_with_cost(&Context::default(), &mut tracker)
            .unwrap();
        assert_eq!(tracker.actual_cost(), 1);
    }

    #[test]
    fn enforces_limit() {
        let program = Program::compile("[1, 2, 3]").unwrap();
        let mut tracker = CostTracker::builder().limit(9).build();
        assert_eq!(
            program.execute_with_cost(&Context::default(), &mut tracker),
            Err(ExecutionError::CostLimitExceeded)
        );
    }
}
