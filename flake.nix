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
                # Fix aba explicit template instanciation
                # Remove this for pinocchio > 3.3.1
                pinocchio = prev.pinocchio.overrideAttrs (super: {
                  patches = (super.patches or [ ]) ++ [
                    (final.fetchpatch {
                      url = "https://github.com/stack-of-tasks/pinocchio/pull/2541/commits/23a638ebfb180aa7d4ea75f17e3d89477dcb6509.patch";
                      hash = "sha256-XIZpq1JK5mY5tv3MqRk/ep6/5cJOjV2gkW1ywLjXUBU=";
                    })
                  ];
                });
                # Ignore pinocchio #2563
                # Remove this for pinocchio > 3.3.1 && crocoddyl >= 3.0.0
                crocoddyl = prev.crocoddyl.overrideAttrs (super: {
                  cmakeFlags =
                    (super.cmakeFlags or [ ])
                    ++ final.lib.optionals final.stdenv.hostPlatform.isDarwin [
                      (final.lib.cmakeFeature "CMAKE_CTEST_ARGUMENTS" "--exclude-regex;test_pybinds_*")
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
