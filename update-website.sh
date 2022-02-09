rm -r /tmp/triton-docs; 
mkdir /tmp/triton-docs;
cp -r CNAME /tmp/triton-docs/
cp -r .nojekyll /tmp/triton-docs/
cp -r update-website.sh /tmp/triton-docs/
cp -r docs/_build/html/* /tmp/triton-docs/
rm -r *
cp -r /tmp/triton-docs/* .
ln -s master docs
git add .
git commit -am "[GH-PAGES] Updated website"