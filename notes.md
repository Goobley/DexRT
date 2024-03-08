Notes
=====

VSCode Env Setup
----------------

- The project is easily portable using devcontainer (on machines with an nvidia GPU). Use the "Reopen folder in devcontainer" if it doesn't launch automatically. Likely need to use the inbuilt terminal to run `cmake_docker.sh` and `make -j 4` for the configuration provider to pick up the necessary compile commands.
- To have the compile options for both the main project and the tests at the same time, use a multi-root workspace:
    - Open a workspace
    - Add the root folder
    - Add the tests folder
    - May need to prompt the configuration provider with scripts/through the `C/C++ Change Configuration Provider` option.