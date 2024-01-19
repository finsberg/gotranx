from pathlib import Path
import json
import gotranx


def main():
    for odepath in Path("odefiles").iterdir():
        if odepath.suffix != ".ode":
            continue
        print(f"Converting {odepath}")
        results_file = Path("expected_ode_output") / odepath.with_suffix(".json").name
        print(f"Writing results to {results_file}")
        ode = gotranx.load_ode(odepath)
        code = gotranx.cli.gotran2py.get_code(ode, scheme=list(gotranx.schemes.Scheme))
        ns = {}
        exec(code, ns)
        parameters = ns["init_parameter_values"]()
        states = ns["init_state_values"]()
        rhs_zero = ns["rhs"](t=0, states=states, parameters=parameters)
        rhs_one = ns["rhs"](t=1, states=states, parameters=parameters)
        print("Evaluate right hand side")
        results = {
            "rhs_0": rhs_zero.tolist(),
            "rhs_1": rhs_one.tolist(),
        }
        for s in gotranx.schemes.Scheme:
            scheme_name = s.value
            print(f"Running scheme {scheme_name}")
            results[scheme_name + "_0"] = ns[scheme_name](
                t=0, states=states, parameters=parameters, dt=0.1
            ).tolist()
            results[scheme_name + "_1"] = ns[scheme_name](
                t=1, states=states, parameters=parameters, dt=0.1
            ).tolist()

        results_file.write_text(json.dumps(results, indent=4))

        # t = np.linspace(0, 1000, 1000)
        # res = solve_ivp(
        #     ns["rhs"], (0, 1000), states, t_eval=t, method="Radau", args=(parmeters,)
        # )
        # try:
        #     v_name = [s.name for s in ode.states if s.name.lower() == "v"][0]
        # except IndexError:
        #     continue
        # v_index = ns["state_index"](v_name)
        # fig, ax = plt.subplots()
        # ax.plot(res.t, res.y[v_index, :])
        # ax.set_title(odepath.name)
        # plt.savefig(results_file.with_suffix(".png"))

        # results = {
        #     "t": res.t.tolist(),
        #     "v": res.y[v_index, :].tolist(),
        # }
        # results_file.write_text(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()
