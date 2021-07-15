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


public class BOW {
    public static class BOWMapper extends Mapper<Object, Text, Text, Text> {
        private Text KEY_OUT = new Text();
        private Text VALUE_OUT = new Text();

        @Override
        // Output　key is filename, value is word#frq
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String word = value.toString().split(" ", 2)[0];
            String files = value.toString().split("\t")[1];
            for(String f_num: files.split(";")) {
                String f_name = f_num.split(": ")[0];
                String num = f_num.split(": ")[1];
                KEY_OUT.set(f_name);
                VALUE_OUT.set(word + "##SEG##" + num);
                context.write(KEY_OUT, VALUE_OUT);
            }
        }
    }


    public static class BOWReducer extends Reducer<Text, Text, Text, Text> {
        private Hashtable idf_table = new Hashtable();
        private int tokenNums = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path [] cacheFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if(cacheFiles != null && cacheFiles.length > 0) {
                String line;
                BufferedReader dataReader = new BufferedReader(new FileReader(cacheFiles[0].toString()));
                try {
                    while((line = dataReader.readLine()) != null) {
                        tokenNums ++;
                        String word = line.split("\t")[0];
                        String idf = line.split("\t")[1];
                        String p = String.valueOf(tokenNums) + ":" + idf;
                        idf_table.put(word, p);
                    }
                } finally {
                    dataReader.close();
                }
            }
        }
        

        private Text VALUE_OUT = new Text();
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            int allnums = 0;
            Hashtable ht = new Hashtable();
            for(Text t: values) {
                String word = t.toString().split("##SEG##")[0];
                int num = Integer.parseInt(t.toString().split("##SEG##")[1]);
                ht.put(word, new Integer(num));
                allnums += num;
            }

            Enumeration terms = ht.keys();
            String out_value = "";
            while(terms.hasMoreElements()) {
                String term = (String)terms.nextElement();
                if(idf_table.containsKey(term)) {
                    int frq = ((Integer)ht.get(term)).intValue();
                    // float tf = (float)frq / (float)allnums;
                    float tf = (float)frq;
                    String s = (String)idf_table.get(term);
                    int idx = Integer.parseInt(s.split(":")[0]) - 1;
                    float idf = Float.parseFloat(s.split(":")[1]);
                    String tf_idf = String.format("%d:%.4f ", idx, tf * idf);
                    out_value += tf_idf;
                }
            }
            out_value = out_value.trim();
            VALUE_OUT.set(out_value);
            context.write(key, VALUE_OUT);
        }
    }

    public static void main(String[] args) {// args[0]: IDF file, args[1]: tokens file, [2]:output
        try {
            Configuration conf = new Configuration();

            DistributedCache.addCacheFile((new Path(args[0])).toUri(), conf);
            Job job = new Job(conf, "BOW");

            job.setJarByClass(BOW.class);

            job.setMapperClass(BOWMapper.class);
            job.setReducerClass(BOWReducer.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(Text.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            FileInputFormat.addInputPath(job, new Path(args[1]));
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}