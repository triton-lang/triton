for i in $(find ../lib/ ../include/isaac/ ../python/src/bind -name '*.cpp' -or -name '*.hpp' -or -name '*.h' | egrep -v "../lib/external");
do
  if ! grep -q Copyright $i
  then
    cat license-header.txt $i >$i.new && mv $i.new $i
  fi
done
