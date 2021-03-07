# mlp_groupwork

Please install:

1. Pytorch

2. Transformer

   ```
   conda install -c huggingface transformers
   ```

3. In order to run on MLP cluster, please:

   ```
   source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
   conda install -c huggingface transformers
   
   conda install -c anaconda importlib-metadata
   
   curl -O http://ftp.gnu.org/gnu/glibc/glibc-2.18.tar.gz
   tar xf  glibc-2.18.tar.gz
   cd glibc-2.18
   mkdir build
   cd build
   ../configure --prefix=/usr
   make -j2
   make install
   ```

4. 