import libsvm.LibSVM;
import net.sf.javaml.core.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileSystem;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

public class SVMtrainer {
    public SVMtrainer(){
    }

    public static class MyMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

        String[] label_rule = new String[]{"财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"};

        public void map(LongWritable key, Text value, Mapper<LongWritable, Text, IntWritable, Text>.Context context) throws IOException, InterruptedException {
            String x_data = "";
            String y_data = "";

            x_data = value.toString().split("\t", 2)[1];
            y_data = value.toString().substring(0, 2);

            for(int i=0; i<14; i++){
                if(y_data.equals(label_rule[i])){
                    String output = "1" + "|" + x_data;
                    //System.out.println(label_rule[i]);
                    context.write(new IntWritable(i), new Text(output));
                }else{
                    String output = "0" + "|" + x_data;
                    //System.out.println(output);
                    context.write(new IntWritable(i), new Text(output));
                }
            }

            //System.out.println("map stage finish");
        }
    }

    public static class MyReducer extends Reducer<IntWritable, Text, NullWritable, Text> {
        int dimensions = 87423;//cn:126408 en:149406 87423

        private ArrayList<String> joinData = new ArrayList();

        protected void ReadLine(String value, Instance tmpInstance) {
            for(String s: value.split(" ")){
                String idx = s.split(":")[0];
                int hash_idx = idx.hashCode() % dimensions;
                double val = Double.parseDouble(s.split(":")[1]) + tmpInstance.value(hash_idx);
                tmpInstance.put(hash_idx, val);
            }
        }


        public void setup(Reducer.Context context) throws IOException, InterruptedException{
                Path[] cacheFiles = context.getLocalCacheFiles();
                if(cacheFiles != null && cacheFiles.length>0){
                    String line;
                    String[] tokens;
                    BufferedReader joinReader = new BufferedReader(new FileReader((cacheFiles[0].toString())));
                    try{
                        while((line = joinReader.readLine())!=null){
                            this.joinData.add(line);
                        }
                    }finally {
                        joinReader.close();
                    }
                }

        }



        public void reduce(IntWritable key, Iterable<Text> values, Reducer<IntWritable, Text, NullWritable, Text>.Context context) throws IOException, InterruptedException {

            LibSVM svm_trainer = new LibSVM();

            Iterator it = values.iterator();
            Dataset data = new DefaultDataset();
            Instance tmpInstance = new SparseInstance(dimensions);


            String line = "";
            String[] buffer = {""};
            int label = 0;
            String vector_str = "";
            int counter = 0;

            while(it.hasNext()) {
                Text value = (Text)it.next();
                    //value: label(in this svm, true or false) | idx:val sparse attribute in the vector
                if(counter%48==0) {
                    line = value.toString();
                    buffer = line.split("\\|");

                    label = Integer.parseInt(buffer[0]);
                    //System.out.println(label);
                    vector_str = buffer[1];


                    if (label == 1) {
                        ReadLine(vector_str, tmpInstance);
                        tmpInstance.setClassValue("positive");
                        //System.out.println(tmpInstance.classValue().toString());

                        data.add(tmpInstance.copy());
                    } else {
                        ReadLine(vector_str, tmpInstance);
                        tmpInstance.setClassValue("negative");
                        //System.out.println(tmpInstance.classValue().toString());

                        data.add(tmpInstance.copy());
                    }
                }
                counter++;
            }

            svm_trainer.buildClassifier(data);

            //Instance check_instance = data.instance(0);


            String y_data = "";
            String test_vector_str = "";
            counter = 0;


            for(String test_str: joinData){
                if(counter%20==0){
                    y_data = test_str.split("\t", 2)[0];
                    test_vector_str = test_str.split("\t", 2)[1];

                    ReadLine(test_vector_str, tmpInstance);

                    double[] scores = svm_trainer.rawDecisionValues(tmpInstance);

                    String scores_str = "";
                    for (double score : scores) {
                        scores_str += Double.toString(score) + " ";
                    }

                    String pre_label_str = svm_trainer.classify(tmpInstance).toString();

                    tmpInstance.clear();

                    String output = y_data + "|" + Integer.toString(key.get()) + "|" + pre_label_str + "|" + scores_str;

                    context.write(NullWritable.get(), new Text(output));
                }
                counter++;

            }

            /*
            //training accuracy
            int correct = 0;
            for(int i=0; i<2000; i++){
                tmpInstance = data.instance(i);
                y_data = svm_trainer.classify(tmpInstance).toString();
                if(y_data.equals(tmpInstance.classValue().toString())){
                    correct++;
                }
            }
            double train_acc = (double) correct/2000;
            context.write(NullWritable.get(), new Text(Double.toString(train_acc)));

             */

            /*
            // weights
            double[] weights = svm_trainer.getWeights();
            String weights_str = "";
            for(double weight: weights){
                weights_str += Double.toString(weight) + " ";
            }
            */

            /*
            // labels
            int[] lalels = svm_trainer.getLabels();
            String label_str = "";
            for(int lab: lalels){
                label_str += Integer.toString(lab) + " ";
            }



            context.write(NullWritable.get(), new Text(label_str));
            */

        }
    }

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();
        //conf.set("mapred.child.java.opts", "-Xmx8192m");
        //conf.set("mapred.reduce.memory.mb", "8192");
        //conf.set("mapred.reduce.child.java.opts", "-Xmx8192m");
        //conf.setInt("dimensions", dimensions);

        DistributedCache.addCacheFile(new Path(args[0]).toUri(), conf);
        Job job = new Job(conf, "SVMtrainer");

        FileSystem fs = FileSystem.get(conf);

        job.setJarByClass(SVMtrainer.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(NullWritable.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(28);

        FileStatus[] package_status = fs.listStatus(new Path(args[1]));
        for(FileStatus f: package_status) {
            job.addFileToClassPath(f.getPath());
        }



        FileInputFormat.addInputPath(job, new Path(args[2]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
