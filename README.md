# mlp_groupwork

Please install:

1. PyTorch （集群上自带，不需要安装）

2. Transformer（本地安装）

   ```
   conda install -c huggingface transformers
   ```

3. Transformer（集群安装）

   ```
   source /home/${STUDENT_ID}/miniconda3/bin/activate mlp
   pip install transformers
   conda install -c anaconda importlib-metadata
   ```

4. 然后测试一下能不能运行：

   ```
   python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
   ```

5. 