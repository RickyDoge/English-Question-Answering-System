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

6. 一些节点的指令：

   ```
   squeue 查看当前job，最小化显示
   smap 查看当前job，详细显示
   sinfo 节点信息
   scanel [job_id] 取消job
   srun -p interactive  --gres=gpu:2 --pty python my_code_exp.py 本地简单测试
   
   文件上传：
   rsync -ua --progress <local_path_of_data_to_transfer> <studentID>@mlp.inf.ed.ac.uk:/home/<studentID>/path/to/folder
   
   文件下载：
   sync -ua --progress <studentID>@mlp.inf.ed.ac.uk:/home/<studentID>/path/to/folder <local_path_of_data_to_transfer>
   
   其它有用的指令：
   squeue -u <user_id> 查看指定用户的任务
   sprio 查看你当前任务的优先级
   scontrol show job <job_id> 查看job的所有信息
   scancel -u <user_id> 取消用户的所有任务
   ```
   
6. 需要先在命令行里手动输入指令，下载模型

   ```
   from transformers import ElectraModel, ElectraTokenizerFast
   tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
   model = ElectraModel.from_pretrained('google/electra-small-discriminator')
   ```

7. 提交任务（使用8块GPU）：

   ```
   sbatch train-standard.sh
   ```

8. 不同的配置

   小型ELECTRA Baseline，在小型数据集上跑（只有squad）

   ```
   python train.py -c baseline-small -d small
   ```

   中型ELECTRA Baseline，在小型数据集上跑（只有squad）

   ```
   python train.py -c baseline-base -d small
   ```

   中型ELECTRA Baseline，在全部数据集上跑（全部数据集，仅留squad-dev做验证集）

   ```
   python train.py -c baseline-base -d normal
   ```

   

