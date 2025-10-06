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
          lib,
          pkgs,
          self',
          ...
        }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          packages = {
            default = self'.packages.py-aligator;
            aligator = pkgs.aligator.overrideAttrs {
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
            py-aligator = pkgs.python3Packages.toPythonModule (self'.packages.aligator.overrideAttrs (super:{
              cmakeFlags = super.cmakeFlags ++ [
                (lib.cmakeBool "BUILD_PYTHON_INTERFACE" true)
                (lib.cmakeBool "BUILD_STANDALONE_PYTHON_INTERFACE" true)
              ];
              nativeBuildInputs = super.nativeBuildInputs ++ [
                pkgs.python3Packages.python
              ];
              propagatedBuildInputs = super.propagatedBuildInputs ++ [
                self'.packages.aligator
                pkgs.python3Packages.crocoddyl
                pkgs.python3Packages.pinocchio
              ];
              nativeCheckInputs = [
                pkgs.ctestCheckHook
                pkgs.python3Packages.pythonImportsCheckHook
              ];
              checkInputs = super.checkInputs++ [
                pkgs.python3Packages.matplotlib
                pkgs.python3Packages.pytest
              ];
              disabledTests = lib.optionals pkgs.stdenv.hostPlatform.isDarwin [
                "aligator-test-py-rollout"
              ];
            }));
          };
        };
    };
}
