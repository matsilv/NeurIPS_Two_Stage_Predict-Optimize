from abc import abstractmethod
import torch
import numpy as np
import gurobipy as gp

from typing import Tuple, Union, Dict, List

PROBLEM_ID = 'stochastic_weights_kp'
VALUES_STR = 'values'
CAPACITY_STR = 'capacity'


class OptimizationProblem:
    """
    Abstract class representing a parameterized optimization problem, as well as a way to solve it.
    """

    def solve_from_torch(self,
                         y_torch: torch.Tensor,
                         opt_prob_params: torch.Tensor,
                         return_runtime: bool = False,
                         **kwargs):
        """
        Solve the problem for a given prediction vector.

        :param y_torch: torch.Tensor; prediction vector.
        :param return_runtime: bool; if True, return also the runtime.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor, float; a vector of decision variable values as a PyTorch Float tensor and the runtime to
                 compute the solution.
        """

        # Convert torch tensors to numpy arrays
        y_numpy = y_torch.detach().numpy()
        opt_prob_params = {key: opt_prob_params[key] for key in opt_prob_params.keys()}

        # Compute the optimal solution and convert to torch tensor
        solution, runtime = self.solve(y_numpy, opt_prob_params, return_runtime=True, **kwargs)
        solution = torch.from_numpy(solution).float()

        if return_runtime:
            return solution, runtime
        else:
            return solution

    @property
    @abstractmethod
    def obj_type(self):
        """
        :return: str; the type of objective function of the optimization problem (linear, quadratic, etc...).
        """

        raise NotImplementedError()

    @abstractmethod
    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the problem for a given prediction vector.
        :param y: numpy.ndarray; the prediction vector.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        raise NotImplementedError()

    @abstractmethod
    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params: Dict) -> Dict:
        """
        Compute the objective value given the predictions and the solution. In this case the predictions are not needed
        to compute the objective value but we want to be consistent with the function signature.
        :param y: torch.Tensor; the predictions.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: dict; we keep track of the total, suboptimality and violation.
        """

        raise NotImplementedError()


class StochasticWeightsKnapsackProblem(OptimizationProblem):
    """
    A class representing a knapsack problem in which the item weights are parameterized.
    """

    def __init__(self, dim: int, penalty: float):
        """
        :param dim: int; the KP dimension.
        :param penalty: float; penalty for the recourse action.
        """

        assert penalty >= 1, "Penalty must be greater than 1, otherwise there is no benefit in preventing the " \
                             "recourse action"

        self._obj_type = 'nonlinear'
        self._is_minimization_problem = False
        self._name = PROBLEM_ID
        self._dim = dim
        self._penalty = penalty

    @property
    def obj_type(self) -> str:
        """
        :return: str; a string identifier of the objective function type.
        """
        return self._obj_type

    @property
    def is_minimization_problem(self) -> bool:
        """
        :return: bool; the optimization problem direction, True for min and False for max.
        """
        return self._is_minimization_problem

    @property
    def name(self) -> str:
        """
        :return: str; the name of the optimization problem.
        """
        return self._name

    @property
    def dim(self) -> int:
        """
        :return: int; the problem dimension (number of items).
        """

        return self._dim

    @staticmethod
    def _convert_opt_prob_params(opt_prob_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        The optimization problem parameters are expected as a dictionary of torch.Tensor. They are converted to
        numpy.ndarray and returned.
        :param opt_prob_params: dict; the keys and values and respectively the names and values of the parameters.
        :return: tuple of np.ndarray; the parameters are returned as a tuple so ordering matters.
        """
        # Unpack the optimization instance-specific parameters
        values = opt_prob_params['values']
        assert isinstance(values, torch.Tensor)
        capacity = opt_prob_params['capacity']
        assert isinstance(capacity, torch.Tensor)

        values = values.numpy()
        capacity = capacity.numpy()

        return values, capacity

    @staticmethod
    def _first_stage_value(values: Union[np.ndarray, torch.Tensor],
                           first_stage_sol: Union[np.ndarray, torch.Tensor, gp.MVar]) -> Union[gp.LinExpr, float]:
        """
        Compute the first stage value. It is simply the matrix multiplication between the item values and the
        first-stage solution.
        :param values: np.ndarray or torch.Tensor; the item values.
        :param first_stage_sol: np.ndarray or torch.Tensor or gp.MVar; the first-stage solution.
        :return: gp.LinExpr or float; the first-stage value as either a Gurobi expression or a numeric value.
        """

        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        return values @ first_stage_sol

    def _second_stage_value(self,
                            values: Union[np.ndarray, torch.Tensor],
                            snd_stage_selected: Union[np.ndarray, torch.Tensor, gp.MVar],
                            snd_stage_removed: Union[np.ndarray, torch.Tensor, gp.MVar]) -> Union[gp.LinExpr, float]:
        """
        Compute the second-stage value. Items selected during second-stage have a lower value. Removing items during
        second-stage incurs in a penalty.
        :param values: np.ndarray or torch.Tensor; the item values.
        :param snd_stage_selected: np.ndarray or torch.Tensor or gp.MVar; the second-stage selected items.
        :param snd_stage_removed: np.ndarray or torch.Tensor or gp.MVar; the second-stage removed items.
        :return: gp.LinExpr or float; the first-stage value as either a Gurobi expression or a numeric value.
        """
        if isinstance(values, torch.Tensor):
            values = values.detach().numpy()

        return 1 / self._penalty * values @ snd_stage_selected - self._penalty * values @ snd_stage_removed

    def solve(self,
              y: np.ndarray,
              opt_prob_params: Dict,
              time_limit: int = 30,
              **kwargs) -> Tuple[np.ndarray, float]:
        """
        Solves the knapsack problem for a given vector of item values.
        :param y: numpy.ndarray; a given vector of item weights.
        :param opt_prob_params: dict; the keys are the parameter names and the values are the corresponding torch.Tensor.
        :param time_limit: int; force a timeout for complex instances.
        :return: numpy.ndarray, float; a vector of decision variable values and the runtime in seconds.
        """

        values, capacity = self._convert_opt_prob_params(opt_prob_params)

        y = np.round(y, decimals=0)
        weights = y

        # Create the Gurobi model
        model = gp.Model()

        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)
        model.setParam('Timelimit', time_limit)

        # If a single weights vector is given then we solve the problem in a deterministic fashion...
        if len(y.shape) == 1:
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")
            model.addConstr(weights @ x <= capacity, name='Packing constraint')
            model.setObjective(self._first_stage_value(values, x), gp.GRB.MAXIMIZE)

        # ...otherwise solve the problem with the Sample Average Approximation algorithm by sampling multiple scenarios
        else:
            # Sanity check: first dimension is the number of scenarios whereas the second on is the number of items
            # (products)
            assert len(y.shape) == 2, "A 2-dimensional array is expected"
            num_scenarios = y.shape[0]
            weights = y

            # First-stage decision variables
            x = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="x")

            # Second-stage decisions
            # Selected items
            u_plus = [model.addMVar(shape=self._dim,
                                    vtype=gp.GRB.BINARY,
                                    name=f"u_plus_{omega}") for omega in range(num_scenarios)]
            # Removed items
            u_minus = [model.addMVar(shape=self._dim,
                                     vtype=gp.GRB.BINARY,
                                     name=f"u_minus_{omega}") for omega in range(num_scenarios)]

            # Initialize the second-stage value
            second_stage_value = 0

            # Add a packing constraints for each scenario
            for omega in range(num_scenarios):

                # Packing constraints
                selected_items_capacity = weights[omega] @ x + \
                                          weights[omega] @ u_plus[omega] - \
                                          weights[omega] @ u_minus[omega]
                model.addConstr(selected_items_capacity <= capacity, name=f"packing_constraints_{omega}")

                # We can only remove already selected items
                model.addConstr(x >= u_minus[omega])

                # We can only add items that have not been selected during first-stage
                model.addConstr(x + u_plus[omega] <= 1)

                # Update the second-stage value
                second_stage_value += self._second_stage_value(values, u_plus[omega], u_minus[omega])

            second_stage_value = 1/num_scenarios * second_stage_value

            # Define the objective function
            model.setObjective(self._first_stage_value(values, x) + second_stage_value, gp.GRB.MAXIMIZE)

        # Solve the model
        model.optimize()

        # Sanity check
        assert model.status in [gp.GRB.TIME_LIMIT, gp.GRB.OPTIMAL]

        # print(model.Runtime)

        return x.x, model.Runtime

    def get_objective_values(self,
                             y: torch.Tensor,
                             sols: torch.Tensor,
                             opt_prob_params) -> Tuple[float, bool]:
        """
        Compute the objective value given the predictions and the solution.
        :param y: torch.Tensor; the item weights.
        :param sols: torch.Tensor; the solutions.
        :param opt_prob_params: torch.Tensor; the instance-specific optimization problem parameters.
        :return: torch.Tensor; the objective value.
        """

        values, capacity = self._convert_opt_prob_params(opt_prob_params)

        sols = sols.numpy()

        y = y.numpy()
        y = np.round(y, decimals=0)

        # Create the Gurobi model
        model = gp.Model()

        # Suppress Gurobi output
        model.setParam('OutputFlag', 0)

        # FIXME: repeating the second-stage optimization problem definition is error-prone: we must ensure it is the
        #  same as the one defined in the "solve" method.
        # Second-stage decisions
        # Selected items
        u_plus = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="u_plus")
        # Removed items
        u_minus = model.addMVar(shape=self._dim, vtype=gp.GRB.BINARY, name="u_minus")

        # Second-stage constraints
        model.addConstr(y @ sols + y @ u_plus - y @ u_minus <= capacity, name=f"packing constraints")

        # We can only remove already selected items
        model.addConstr(sols >= u_minus)

        # We can only add items that have not been selected during first-stage
        model.addConstr(sols + u_plus <= 1)

        # Define the objective function
        tot_value = self._first_stage_value(values, sols) + self._second_stage_value(values, u_plus, u_minus)
        model.setObjective(tot_value, gp.GRB.MAXIMIZE)

        # Solve the model
        model.optimize()

        # Sanity check
        assert model.status == gp.GRB.OPTIMAL

        suboptimality_cost = self._first_stage_value(values, sols)
        penalty_cost = self._second_stage_value(values, u_plus.x, u_minus.x)

        if penalty_cost != 0:
            feasible = False
        else:
            feasible = True

        # Return the cost as a dictionary with information about the total cost, the cost of the feasible solutions
        # and the ration of feasible solutions
        cost = suboptimality_cost + penalty_cost

        return cost, feasible


def correction_single_obj(realPrice,
                          predPrice,
                          cap,
                          realWeightTemp,
                          predWeightTemp,
                          penalty: float,
                          item_num: int) -> Tuple[float, List[int]]:

    objective = None
    sol = []

#    print("realPrice: ", realPrice, "predPrice: ", predPrice)
    realWeight = np.zeros(item_num)
    predWeight = np.zeros(item_num)
    realPriceNumpy = np.zeros(item_num)
    for i in range(item_num):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]

    # Why should we ignore solutions for negative weights?
    # if min(predWeight) >= 0:
    predWeight = predWeight.tolist()

    # Stage 1:
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(item_num, vtype=gp.GRB.BINARY, name='x')
    # m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
    m.setObjective(x.prod(predPrice), gp.GRB.MAXIMIZE)
    m.addConstr((x.prod(predWeight)) <= cap)

    m.optimize()
    predSol = np.zeros(item_num,dtype='i')
    x1_selecteditem_num = 0
    for i in range(item_num):
        predSol[i] = x[i].x
        if x[i].x == 1:
          x1_selecteditem_num = x1_selecteditem_num + 1
    objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

    # Stage 2:
    realWeight = realWeight.tolist()
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    x = m2.addVars(item_num, vtype=gp.GRB.BINARY, name='x')
    snd_stage_removed = m2.addVars(item_num, vtype=gp.GRB.BINARY, name='2nd_stage_removed')
    snd_staged_selected = m2.addVars(item_num, vtype=gp.GRB.BINARY, name='2nd_stage_selected')

    # OBJ = purchase_fee * x.prod(realPrice)
    OBJ = x.prod(realPrice)
    for i in range(item_num):
        OBJ = OBJ - penalty * realPrice[i] * snd_stage_removed[i] + 1/penalty * realPrice[i] * snd_staged_selected[i]
    m2.setObjective(OBJ, gp.GRB.MAXIMIZE)

    m2.addConstr((x.prod(realWeight) - snd_stage_removed.prod(realWeight) + snd_staged_selected.prod(realWeight)) <= cap)
    for i in range(item_num):
        m2.addConstr(x[i] == predSol[i])
        m2.addConstr(x[i] >= snd_stage_removed[i])
        m2.addConstr(x[i] + snd_staged_selected[i] <= 1)

    try:
        m2.optimize()
        objective = m2.objVal
        sol = []
        x2_selecteditem_num = 0
        for i in range(item_num):
            sol.append(x[i].x - snd_stage_removed[i].x + snd_staged_selected[i].x)
            if x[i].x - snd_stage_removed[i].x + snd_staged_selected[i].x == 1:
              x2_selecteditem_num = x2_selecteditem_num + 1
#        print("Stage 2: ", sol, objective)
    except:
        print(predPrice, predWeight, realPrice, realWeight, predSol)

    return objective, sol
