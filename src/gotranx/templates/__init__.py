from __future__ import annotations
import typing

from . import c
from . import python
from . import jax
from . import julia


class Template(typing.Protocol):
    @staticmethod
    def state_index(data: dict[str, int]) -> str:
        """The state_index function is a function that returns
        the index of the state with the given name.

        Parameters
        ----------
        data : dict[str, int]
            The data containing the state names and their indexes

        Returns
        -------
        str
            The code for the state_index function
        """

    @staticmethod
    def parameter_index(data: dict[str, int]) -> str:
        """The parameter_index function is a function that returns
        the index of the parameter with the given name.

        Parameters
        ----------
        data : dict[str, int]
            The data containing the parameter names and their indexes

        Returns
        -------
        str
            The code for the parameter_index function
        """

    @staticmethod
    def monitor_index(data: dict[str, int]) -> str:
        """The monitor_index function is a function that returns
        the index of the monitor with the given name.

        A monitor is a variable that is not part of the state or parameter
        but is used to monitor the simulation. For example, in a cardiac
        cell models, different currents across the membrane are examples
        of monitors.

        Parameters
        ----------
        data : dict[str, int]
            The data containing the monitor names and their indexes

        Returns
        -------
        str
            The code for the monitor_index function
        """

    @staticmethod
    def missing_index(data: dict[str, int]) -> str:
        """The missing_index function is a function that returns
        the index of the missing value with the given name.

        A missing value is a variable that is not part of the state or parameter
        but is used in the right-hand side of the ODE. This typically happens
        when the model is not fully defined, and some variables are missing, for
        example when you split an ODE into two sub-models and variables needs
        to be passed between them.

        Parameters
        ----------
        data : dict[str, int]
            The data containing the missing value names and their indexes

        Returns
        -------
        str
            The code for the missing_index function
        """

    @staticmethod
    def init_state_values(
        name: str, state_values: list[float], state_names: list[str], code: str
    ) -> str:
        """The init_state_values function is a function that initializes
        the state values of the model.

        The function typically return an array of the state values, or in
        the case of C code, it takes a pointer to the array of state values
        and initializes the values in place.

        Parameters
        ----------
        name : str
            Name of the return variable
        state_values : list[float]
            Default values for the states
        state_names : list[str]
            The names of the states
        code : str
            Additional code to be inserted into the function

        Returns
        -------
        str
            The code for the init_state_values function
        """

    @staticmethod
    def init_parameter_values(
        name: str, parameter_values: list[float], parameter_names: list[str], code: str
    ) -> str:
        """The init_parameter_values function is a function that initializes
        the parameter values of the model.

        The function typically return an array of the parameter values, or in
        the case of C code, it takes a pointer to the array of parameter values
        and initializes the values in place.

        Parameters
        ----------
        name : str
            Name of the return variable
        parameter_values : list[float]
            Default values for the parameters
        parameter_names : list[str]
            The names of the parameters
        code : str
            Additional code to be inserted into the function

        Returns
        -------
        str
            The code for the init_parameter_values function
        """

    @staticmethod
    def method(
        name: str,
        args: str,
        states: str,
        parameters: str,
        values: str,
        return_name: str | None,
        num_return_values: int,
        shape_info: str,
        values_type: str,
        missing_variables: str,
        post_function_signature: str,
    ) -> str:
        """The method function is a function that generates a method
        for the model.

        One example of a method is the right-hand side of the ODE, which
        calculates the derivatives of the states. Another example is the
        monitor values, which calculates the values of the monitors.
        All the different numerical schemes are also examples of methods.


        Parameters
        ----------
        name : str
            The name of the method
        args : str
            The arguments of the method
        states : str
            The code for assigning the states
        parameters : str
            The code for assigning the parameters
        values : str
            The code for assigning the values
        return_name : str | None
            The name of the return variable
        num_return_values : int
            The number of return values
        shape_info : str
            The shape information of the return values
        values_type : str
            The type of the values
        missing_variables : str
            The code for handling missing variables
        post_function_signature : str
            The code going after the function signature (e.g. 'where T' or '-> None')

        Returns
        -------
        str
            The code for the method
        """


__all__ = ["c", "python", "jax", "julia", "Template"]
