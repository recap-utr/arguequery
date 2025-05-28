{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs =
    inputs@{
      flake-parts,
      systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem =
        {
          pkgs,
          lib,
          config,
          ...
        }:
        {
          devShells.default = pkgs.mkShell {
            LD_LIBRARY_PATH = lib.makeLibraryPath [
              pkgs.stdenv.cc.cc
              pkgs.zlib
              "/run/opengl-driver"
            ];
            packages = with pkgs; [ uv ];
            UV_PYTHON = lib.getExe pkgs.python312;
            TOKENIZERS_PARALLELISM = true;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
          devShells.orbstack = config.devShells.default.overrideAttrs (oldAttrs: {
            UV_LINK_MODE = "copy";
            UV_PREVIEW = true; # https://github.com/astral-sh/uv/issues/11819
            UV_PROJECT_ENVIRONMENT = ".venv-orbstack";
          });
        };
    };
}
