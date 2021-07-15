import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.util.Iterator;


public class SVMpredictor {
    public SVMpredictor(){
    }


    public static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {


        public void map(LongWritable key, Text value, Mapper<LongWritable, Text, Text, Text>.Context context) throws IOException, InterruptedException {
            //System.out.println(value.toString());
            String[] buffer = value.toString().split("\\|",2);
            String file_name = buffer[0];
            String output = buffer[1];

            context.write(new Text(file_name), new Text(output));
        }
    }

    public static class MyReducer extends Reducer<Text, Text, Text, Text> {

        String[] label_rule = new String[]{"财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"};

        public void reduce(Text key, Iterable<Text> values, Reducer<Text, Text, Text, Text>.Context context) throws IOException, InterruptedException {
            Iterator it = values.iterator();

            String label = key.toString().substring(0, 2);

            double[] scores = new double[14];
            String[] pred_true_false = new String[14];

            while(it.hasNext()){
                Text value = (Text)it.next();
                //System.out.println(value.toString());
                String[] buffer = value.toString().split("\\|");
                int svm_num = Integer.parseInt(buffer[0]);
                String pred_label_str = buffer[1];
                double score = Double.parseDouble(buffer[2]);
                scores[svm_num] = score;
                pred_true_false[svm_num] = pred_label_str;
                //System.out.println(pred_label_str);
            }

            double max_score = scores[0];
            int pred_label_index = 0;
            for(int i=1; i<14; i++) {
                    if (scores[i] < max_score) {
                        pred_label_index = i;
                        max_score = scores[i];
                    }
            }

            /*
            if(label.equals(label_rule[pred_label_index])){
                context.write(key, new Text(Integer.toString(1)));
            }else{
                context.write(key, new Text(Integer.toString(0)));
            }
             */
            context.write(key, new Text(Integer.toString(pred_label_index)));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();

        Job job = new Job(conf, "SVMpredictor");
        job.setJarByClass(SVMpredictor.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}
