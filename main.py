
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from plots.simulation_average_quorum_time import run_simulation_over_all_behaviour, Dataset


def main(
        dataset: Annotated[
            Dataset, typer.Option("--data", "-d", case_sensitive=False,
                                  help="Dataset which will be used to simulate latencies in the network")
        ] = Dataset.WONDER,
        output_directory: Annotated[
            Optional[Path], typer.Option("--out", "-o", exists=True, help="Output directory for the plots")] = ".",
        is_multiprocess: Annotated[bool, typer.Option("--multiprocess","-p",
                                                      help="Experimental feature in which program is runned in multiprocess fashion")] = False,
):
    run_simulation_over_all_behaviour(str(output_directory / 'Comparison of Quorum Reaching Time with '), dataset,
                                      is_multiprocess)


if __name__ == "__main__":
    typer.run(main)
