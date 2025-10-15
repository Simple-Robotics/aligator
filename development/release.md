# Release with Pixi

To create a release with Pixi run the following commands on the **main** branch:

```bash
ALIGATOR_VERSION=X.Y.Z pixi run release_new_version
git push origin
git push origin vX.Y.Z
```

Where `X.Y.Z` is the new version.
Be careful to follow the [Semantic Versioning](https://semver.org/spec/v2.0.0.html) rules.

You will find the following assets:
- `./build_new_version/aligator-X.Y.Z.tar.gz`
- `./build_new_version/aligator-X.Y.Z.tar.gz.sig`

Then, create a new release on [GitHub](https://github.com/Simple-Robotics/aligator/releases/new) with:

* Tag: vX.Y.Z
* Title: aligator vX.Y.Z
* Body:
```
## What's Changed

CHANGELOG CONTENT

**Full Changelog**: https://github.com/Simple-Robotics/aligator/compare/vXX.YY.ZZ...vX.Y.Z
```

Where `XX.YY.ZZ` is the last release version.

Then upload `aligator-X.Y.Z.tar.gz` and `aligator-X.Y.Z.tar.gz.sig` and publish the release.
