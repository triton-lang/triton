for i in $(find ../lib/ ../include/isaac/ ../python/src/bind -name '*.cpp' -or -name '*.hpp' -or -name '*.h' | grep -v "../lib/external" | grep -v "../include/isaac/driver/external/");
do
  if ! grep -q Copyright $i
  then
    cat ../LICENSE $i >$i.new && mv $i.new $i
  fi
done
