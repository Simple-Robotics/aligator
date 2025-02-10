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
          system,
          ...
        }:
        {
          _module.args.pkgs = import inputs.nixpkgs {
            inherit system;
            overlays = [
              (final: prev: {
                pinocchio = prev.pinocchio.overrideAttrs (super: {
                  patches = (super.patches or [ ]) ++ [
                    (final.fetchpatch {
                      url = "https://github.com/stack-of-tasks/pinocchio/pull/2541/commits/23a638ebfb180aa7d4ea75f17e3d89477dcb6509.patch";
                      hash = "sha256-XIZpq1JK5mY5tv3MqRk/ep6/5cJOjV2gkW1ywLjXUBU=";
                    })
                  ];
                });
              })
            ];
          };
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
