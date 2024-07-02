from ctypes import c_char_p
from ctypes import c_double
from ctypes import c_int
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np


class CModel:
    def __init__(self, lib, ode):
        self.lib = lib

        # Get number of states and parameters from the C library
        self.num_states = self.lib.state_count()
        self.num_parameters = self.lib.parameter_count()
        self.num_monitored = self.lib.monitor_count()
        self._init_lib()
        self.ode = ode

    def __repr__(self) -> str:
        return f"CModel({self.lib}, {self.ode})"

    def __str__(self) -> str:
        return (
            f"Model named {self.ode.name} with {self.num_states} states, "
            f"{self.num_parameters} parameters and {self.num_monitored} "
            "monitored values"
        )

    def parameter_values_to_dict(
        self,
        parameter_values: np.ndarray,
    ) -> Dict[str, float]:
        """Convert the parameter values using the ordered from the C library
        to a dictionary with parameter names as keys and the values as values"""
        names = self.parameter_names
        values = [parameter_values[self.parameter_index(name)] for name in names]
        return dict(zip(names, values))

    def state_values_to_dict(self, state_values: np.ndarray) -> Dict[str, float]:
        """Convert the state values using the ordered from the C library
        to a dictionary with state names as keys and the values as values"""
        names = self.state_names
        values = [state_values[self.state_index(name)] for name in names]
        return dict(zip(names, values))

    def parameter_dict_to_array(self, parameter_dict: Dict[str, float]) -> np.ndarray:
        """Convert the a dictionary of parameters to an array of values
        with the correct order.
        """
        values = self.initial_parameter_values()
        for name, value in parameter_dict.items():
            values[self.parameter_index(name)] = value
        return values

    def state_dict_to_array(self, state_dict: Dict[str, float]) -> np.ndarray:
        """Convert the a dictionary of states to an array of values
        with the correct order.
        """
        values = self.initial_state_values()
        for name, value in state_dict.items():
            values[self.state_index(name)] = value
        return values

    def default_parameters(self) -> Dict[str, float]:
        """Return the default parameter as a dictionary where the
        keys are the parameter names and the values"""
        return self.parameter_values_to_dict(self.initial_parameter_values())

    def default_initial_states(self) -> Dict[str, float]:
        """Return the default initial as a dictionary where the
        keys are the parameter names and the values"""
        return self.state_values_to_dict(self.initial_state_values())

    @property
    def parameter_names(self) -> List[str]:
        """List of parameters names"""
        return [p.name for p in self.ode.parameters]

    @property
    def state_names(self) -> List[str]:
        """List of state names"""
        return [p.name for p in self.ode.states]

    @property
    def monitor_names(self) -> List[str]:
        """List of monitor names"""
        return [expr.name for expr in self.ode.intermediates + self.ode.state_derivatives]

    def _init_lib(self):
        """
        Make sure that arrays passed to C is of the correct types.
        """

        float64_array = np.ctypeslib.ndpointer(
            dtype=c_double,
            ndim=1,
            flags="contiguous",
        )
        int32_array = np.ctypeslib.ndpointer(
            dtype=c_int,
            ndim=1,
            flags="contiguous",
        )
        float64_array_2d = np.ctypeslib.ndpointer(
            dtype=c_double,
            ndim=2,
            flags="contiguous",
        )

        self.lib.init_state_values.restype = None  # void
        self.lib.init_state_values.argtypes = [float64_array]

        self.lib.init_parameter_values.restype = None  # void
        self.lib.init_parameter_values.argtypes = [float64_array]

        self.lib.state_index.restype = c_int
        self.lib.state_index.argtypes = [c_char_p]  # state_name

        self.lib.parameter_index.restype = c_int
        self.lib.parameter_index.argtypes = [c_char_p]  # state_name

        self.lib.monitor_index.restype = c_int
        self.lib.monitor_index.argtypes = [c_char_p]  # state_name

        self.lib.monitored_values.restype = None
        self.lib.monitored_values.argtypes = [
            float64_array_2d,  # monitored
            float64_array_2d,  # states
            float64_array,  # parameters
            float64_array,  # u
            float64_array,  # t_values
            c_int,  # num_timesteps
            int32_array,  # indices
            c_int,  # num_indices
        ]

        advance_functions = [
            self.lib.forward_explicit_euler,
            self.lib.forward_generalized_rush_larsen,
        ]

        for func in advance_functions:
            func.restype = None  # void
            func.argtypes = [
                float64_array,  # u
                c_double,  # t
                c_double,  # dt
                float64_array,  # parameters
            ]

        solve_functions = [
            self.lib.ode_solve_forward_euler,
            self.lib.ode_solve_rush_larsen,
        ]

        for func in solve_functions:
            func.restype = None  # void
            func.argtypes = [
                float64_array,  # u
                float64_array,  # parameters
                float64_array_2d,  # u_values
                float64_array,  # t_values
                c_int,  # num_timesteps
                c_double,  # dt
            ]

    def advance_ODEs(
        self,
        states: np.ndarray,
        t: float,
        dt: float,
        parameters: np.ndarray,
        scheme="GRL1",
    ) -> np.ndarray:
        u = states.copy()
        if scheme == "GRL1":
            self.lib.forward_generalized_rush_larsen(u, t, dt, parameters)
        elif scheme == "FE":
            self.lib.forward_explicit_euler(u, t, dt, parameters)
        else:
            raise ValueError(f"Unknown scheme {scheme}")

        return u

    def monitor(
        self,
        names: list[str],
        states: np.ndarray,
        t: np.ndarray,
        parameters: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Return a single monitored value

        Parameters
        ----------
        names : list[str]
            Names of monitored values
        states : np.ndarray
            The states
        t : np.ndarray
            The time steps
        parameters : Dict[str, float], optional
            Dictionary with initial parameters, by default None

        Returns
        -------
        np.ndarray
            The values of the monitors
        """
        indices = np.array([self.monitor_index(name) for name in names], dtype=np.int32)

        parameter_values = self._get_parameter_values(parameters=parameters)
        u = np.zeros(self.num_states, dtype=np.float64)
        monitored_values = np.zeros((t.size, len(names)), dtype=np.float64)

        self.lib.monitored_values(
            monitored_values,
            states,
            parameter_values,
            u,
            t,
            t.size,
            indices,
            len(indices),
        )
        return monitored_values

    def state_index(self, state_name: str) -> int:
        """Given a name of a state, return the index of it.

        Arguments
        ---------
        state_name : str
            Name of the state

        Returns
        -------
        int
            The index of the given state

        Note
        ----
        To list all possible states see `BaseModel.state_names`

        """
        assert isinstance(state_name, str)
        if state_name not in self.state_names:
            raise ValueError(f"Invalid state name {state_name!r}")

        state_name_bytestring = state_name.encode()
        return self.lib.state_index(state_name_bytestring)

    def parameter_index(self, parameter_name: str) -> int:
        """Given a name of a parameter, return the index of it.

        Arguments
        ---------
        parameter_name : str
            Name of the parameter

        Returns
        -------
        int
            The index of the given parameter

        """
        assert isinstance(parameter_name, str)
        if parameter_name not in self.parameter_names:
            raise ValueError(f"Invalid parameter name {parameter_name!r}")

        parameter_name_bytestring = parameter_name.encode()
        return self.lib.parameter_index(parameter_name_bytestring)

    def monitor_index(self, monitor_name: str) -> int:
        """Given a name of a monitored expression, return the index of it.

        Arguments
        ---------
        monitor_name : str
            Name of the monitored expression

        Returns
        -------
        int
            The index of the given monitored expression


        """
        assert isinstance(monitor_name, str)
        if monitor_name not in self.monitor_names:
            raise ValueError(f"Invalid monitor name {monitor_name!r}")

        monitor_name_bytestring = monitor_name.encode()
        return self.lib.monitor_index(monitor_name_bytestring)

    def initial_parameter_values(self) -> np.ndarray:
        """Return the default parameters as a numpy array"""
        parameters = np.zeros(self.num_parameters, dtype=np.float64)
        self.lib.init_parameter_values(parameters)
        return parameters

    def initial_state_values(self, **values) -> np.ndarray:
        """Return the default initial states as a numpy array"""
        states = np.zeros(self.num_states, dtype=np.float64)
        self.lib.init_state_values(states)

        for key, value in values.items():
            states[self.state_index(key)] = value

        return states

    def _get_parameter_values(
        self,
        parameters: Optional[Dict[str, float]],
        verbose: bool = False,
    ) -> np.ndarray:
        parameter_values = self.initial_parameter_values()
        if parameters is not None:
            assert isinstance(parameters, dict)
            for name, new_value in parameters.items():
                index = self.parameter_index(name)
                old_value = parameter_values[index]
                if old_value != new_value:
                    parameter_values[index] = new_value
                    if verbose:
                        print(
                            f"Update parameter {name} from " f"{old_value} to {new_value}",
                        )
        return parameter_values

    def solve(
        self,
        t_start: float,
        t_end: float,
        dt: float,
        num_steps: Optional[int] = None,
        method: str = "GRL1",
        u0: Optional[np.ndarray] = None,
        parameters: Optional[Dict[str, float]] = None,
        verbose: bool = False,
    ):
        """Solve the model

        Parameters
        ----------
        t_start : float
            Time at start point
        t_end : float
            Time at end point
        dt : float
            Time step for solver
        num_steps : Optional[int], optional
            Number of steps to use, by default None
        method : str, optional
            Scheme for solving the ODE. Options are
            'GRL1' (first order generalized Rush Larsen) or
            'FE' (forward euler), by default "GRL1"
        u0 : Optional[np.ndarray], optional
            Initial state variables. If none is provided then
            the default states will be used, by default None.
        parameters : Optional[Dict[str, float]], optional
            Parameter for the model. If none is provided then
            the default parameters will be used, by default None.
        verbose : bool, optional
            Print more output, by default False

        """
        parameter_values = self._get_parameter_values(
            parameters=parameters,
            verbose=verbose,
        )

        if not isinstance(dt, float):
            dt = float(dt)
        if num_steps is not None:
            assert isinstance(num_steps, int)
            t_end = dt * num_steps
        else:
            num_steps = round((t_end - t_start) / dt)

        t_values = np.linspace(t_start, t_end, num_steps + 1)

        if u0 is not None:
            assert len(u0) == self.num_states
        else:
            u0 = np.zeros(self.num_states, dtype=np.float64)
            self.lib.init_state_values(u0)

        u_values = np.zeros((num_steps + 1, u0.shape[0]), dtype=np.float64)
        u_values[0, :] = u0[:]

        if method == "FE":
            self.lib.ode_solve_forward_euler(
                u0,
                parameter_values,
                u_values,
                t_values,
                num_steps,
                dt,
            )
        elif method == "GRL1":
            self.lib.ode_solve_rush_larsen(
                u0,
                parameter_values,
                u_values,
                t_values,
                num_steps,
                dt,
            )
        else:
            raise ValueError(f"Invalid method {method}")

        return Solution(
            time=t_values,
            u=u_values,
            parameter_values=parameter_values,
            model=self,
        )


class Solution(NamedTuple):
    time: np.ndarray
    u: np.ndarray
    parameter_values: np.ndarray
    model: CModel

    @property
    def parameters(self):
        return self.model.parameter_values_to_dict(self.parameter_values)

    def keys(self):
        return self.model.state_names

    def monitor_keys(self):
        return self.model.monitor_names

    def monitor(self, names: list[str]):
        return self.model.monitor(
            names,
            self.u,
            self.time,
            parameters=self.parameters,
        )

    def __getitem__(self, name):
        if name not in self.keys():
            raise KeyError("Key {name} not a valid state name")
        index = self.model.state_index(name)
        return self.u[:, index]
