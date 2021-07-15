// 
import java.util.*;
import java.io.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Job;

import org.apache.hadoop.filecache.DistributedCache;

import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;

public class IDF {
    public static class IDFMapper extends Mapper<Object, Text, Text, FloatWritable> {
        private int filesnum = 0;
        private Text out_key = new Text();
        private FloatWritable out_value = new FloatWritable();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            filesnum = conf.getInt("filesnum", 0);
        }

        @Override
        // output: word idf 
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String word = value.toString().split(" ", 2)[0];
            
            String s_num = value.toString().split(",", 2)[0];
            int word_num = Integer.parseInt(s_num.split(" ")[1]);
            if(word_num > 10) {
                // remove words whose length < 2
                if(word.length() > 3 && word.substring(1, word.length() - 1).matches("[\u4E00-\u9FA5]+")) {// chinese
                //if(word.length() > 3) { 
                    String files = value.toString().split("\t")[1];
                    int nums = files.split(";").length;
                    out_key.set(word);
                    float idf = (float)Math.log((float)filesnum / (float)(nums + 1));
                    out_value.set(idf);
                    context.write(out_key, out_value);
                }
            }
        }
    }


    public static class IDFReducer extends Reducer<Text, FloatWritable, Text, FloatWritable> {
        @Override
        protected void reduce(Text key, Iterable<FloatWritable> values, Context context) throws IOException, InterruptedException {
            for(FloatWritable f: values) {
                context.write(key, f);
            }
        }
    }

    public static void main(String[] args) {// args[0]: .../3-classification/ args[1]:tokenize args[2]:output
        try {
            Configuration conf = new Configuration();

            int filesnum = 0;
            FileSystem fs = FileSystem.get(conf);
            FileStatus[] status = fs.listStatus(new Path(args[0] + "train"));
            for(FileStatus f: status) { 
                if(f.isDir()) {
                    FileStatus[] son_status = fs.listStatus(f.getPath());
                    filesnum += son_status.length;
                    System.out.println(filesnum);
                } 
            }
            conf.setInt("filesnum", filesnum);

            Job job = new Job(conf, "IDF");
            job.setJarByClass(IDF.class);

            job.setMapperClass(IDFMapper.class);
            job.setReducerClass(IDFReducer.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(FloatWritable.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(FloatWritable.class);

            FileInputFormat.addInputPath(job, new Path(args[1]));
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}