{
  description = "Versatile and efficient framework for constrained trajectory optimization";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        overrideAttrs.aligator = {
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./bench
              ./bindings
              ./CMakeLists.txt
              ./extra-python-macros.cmake
              ./doc
              ./examples
              ./include
              ./package.xml
              ./src
              ./tests
            ];
          };
        };
      }
    );
}
