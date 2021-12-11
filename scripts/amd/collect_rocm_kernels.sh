# COPY kernels
DIRNAME=triton_rocm_kernels
rm -rf $DIRNAME
mkdir $DIRNAME
mv /tmp/*.ttir $DIRNAME
mv /tmp/*.ll $DIRNAME
mv /tmp/*.gcn $DIRNAME
mv /tmp/*.o $DIRNAME
mv /tmp/*.hsaco $DIRNAME
mv /tmp/*.s $DIRNAME
chmod -R 777 $DIRNAME
