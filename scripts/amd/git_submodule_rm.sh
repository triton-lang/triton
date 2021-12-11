SUBMODULE=third_party/hipify-torch

# Remove the submodule entry from .git/config
git submodule deinit -f $SUBMODULE

# Remove the submodule directory from the superproject's .git/modules directory
rm -rf .git/modules/$SUBMODULE

# Remove the entry in .gitmodules and remove the submodule directory located at path/to/submodule
git rm -f $SUBMODULE