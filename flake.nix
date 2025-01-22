{
  description = "Versatile and efficient framework for constrained trajectory optimization";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        {
          pkgs,
          self',
          ...
        }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.aligator;
            aligator = pkgs.python3Packages.aligator.overrideAttrs {
              src = pkgs.lib.fileset.toSource {
                root = ./.;
                fileset = pkgs.lib.fileset.unions [
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
                  ./gar
                ];
              };
            };
          };
        };
    };
}
