import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.Iterator;

public class Accuracy {
    public Accuracy(){
    }



    public static class MyMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable>{

        String[] label_rule = new String[]{"财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"};

        public void map(LongWritable key, Text value, Mapper<LongWritable, Text, IntWritable, IntWritable>.Context context) throws IOException, InterruptedException {
            String filename = value.toString().split("\t")[0].substring(0,2);
            String pred_res = value.toString().split("\t")[1];

            if (filename.equals(label_rule[Integer.parseInt(pred_res)])) {
                context.write(new IntWritable(0), new IntWritable(1));
            }else{
                context.write(new IntWritable(0), new IntWritable(0));
            }
        }
    }

    public static class MyReducer extends Reducer<IntWritable, IntWritable, DoubleWritable, NullWritable> {
        public void reduce(IntWritable key, Iterable<IntWritable> values, Reducer<IntWritable, IntWritable, DoubleWritable, NullWritable>.Context context) throws  IOException, InterruptedException {
            Iterator it = values.iterator();

            long total = 0;
            long right = 0;

            while(it.hasNext()){
                IntWritable value = (IntWritable) it.next();

                int val = value.get();

                total += 1;
                right += val;
            }

            double accuracy = (double) right/total * 100;
            context.write(new DoubleWritable(accuracy), NullWritable.get());
        }
    }

    public static void main(String[] args) throws Exception{
        Configuration conf = new Configuration();

        Job job = new Job(conf, "Accuracy");
        job.setJarByClass(Accuracy.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(DoubleWritable.class);
        job.setOutputValueClass(NullWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
