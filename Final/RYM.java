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

public class RYM {
    public static class RYMMapper extends Mapper<Object, Text, Text, FloatWritable> {
        private int dimensions = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Configuration conf = context.getConfiguration();
            dimensions = conf.getInt("dimensions", 0);
        }

        float[] x_data = new float[dimensions];
        String y_data = "";
        @Override 
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            y_data = pmzReadLine(value, x_data);
        }

        protected static String pmzReadLine(Text value, float[] x_data) {
            String raw_x = value.toString().split("\t", 2)[1];
            for(String s: raw_x.split(" ")) {
                int idx = Integer.parseInt(s.split(":")[0]);
                float val = Float.parseFloat(s.split(":")[1]);
                x_data[idx] = val;
            }
            return value.toString().substring(0, 2);
        }
    }


    public static void main(String[] args) {
        try {
            Configuration conf = new Configuration();

            int dimensions = 126408;//cn:126408 en:149406
            conf.setInt("dimensions", dimensions);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}