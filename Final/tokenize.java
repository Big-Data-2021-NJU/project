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

import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;


public class tokenize {
    public static class TkMapper extends Mapper<Object, Text, Text, IntWritable> {
        private HashSet<String> stopwords = new HashSet<String>();
        private Text term_file = new Text();
        private IntWritable frq = new IntWritable();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path [] cacheFiles = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            if(cacheFiles != null && cacheFiles.length > 0) {
                String line;
                BufferedReader dataReader = new BufferedReader(new FileReader(cacheFiles[0].toString()));
                try {
                    while((line = dataReader.readLine()) != null) {
                        stopwords.add(line);
                    }
                } finally {
                    dataReader.close();
                }
            }
        }

        @Override
        // Output　key is term#SEG#filename: frequency
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            List<String> SplitedLine = segment(value.toString());
            Hashtable ht = new Hashtable();
            
            if(SplitedLine != null) {
                for(String s: SplitedLine) {
                    s = s.replaceAll("(\\d+|[,+$&，_#.-])","");// remove digits and .-
                    if(s != "" && !stopwords.contains(s)) {// remove the stopwrods
                        if(!ht.containsKey(s)) ht.put(s, new Integer(1));
                        else {
                            int wc = ((Integer)ht.get(s)).intValue() + 1;
                            ht.put(s, new Integer(wc));
                        }
                    }
                }
            }

            FileSplit filesplit = (FileSplit)context.getInputSplit();
            String filename = filesplit.getPath().getName();

            Enumeration terms = ht.keys();
            while(terms.hasMoreElements()) {
                String term = (String)terms.nextElement();
                term_file.set(term + "#SEG#" + filename);
                frq.set(((Integer)ht.get(term)).intValue());
                context.write(term_file, frq);
            }
        }

        // reference: https://vimsky.com/examples/detail/java-class-org.wltea.analyzer.core.IKSegmenter.html
        private List<String> segment(String str) throws IOException, InterruptedException {
            if(str == null || str.trim().equals("")) return null;
            
            InputStream is = new ByteArrayInputStream(str.getBytes());
            IKSegmenter iks = new IKSegmenter(new InputStreamReader(is), true);//true is use smart
            Lexeme lexeme = null;
            List<String> list = new ArrayList<String>();
            while((lexeme = iks.next()) != null) {
                list.add(lexeme.getLexemeText());
            }
            return list;
        }
    }

    public static class TkPartitioner extends HashPartitioner<Text,IntWritable> {
        @Override
        public int getPartition(Text key, IntWritable value, int numReduceTasks) {
            Text term = new Text(key.toString().split("#SEG#")[0]);
            return super.getPartition(term, value, numReduceTasks);
        }
    }


    public static class TkReducer extends Reducer<Text, IntWritable, Text, Text> {
        private String prev = "";
        private List<String> postingsList = new LinkedList<String>();
        private int total_sum = 0;
        private Text OUT_KEY = new Text();
        private Text OUT_VALUE = new Text();

        private int threshold = 4;// words less than 2 will not be contained in dictionary
        
        @Override
        // Output: term totalnum    file1:num1,file2:num2
        protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            String cur_term = key.toString().split("#SEG#")[0];
            String cur_file = key.toString().split("#SEG#")[1];
            Iterator<IntWritable> it = values.iterator();
            if(prev.length() != 0 && !cur_term.equals(prev)) {
                if(total_sum >= threshold) {
                    String out_k = String.format("[%s] %d,", prev, total_sum);
                    OUT_KEY.set(out_k);

                    String out_v = "";
                    for(String s: postingsList) out_v += s;
                    OUT_VALUE.set(out_v.substring(0, out_v.length() - 1));
                    context.write(OUT_KEY, OUT_VALUE);// remove the last ';'
                    total_sum = 0;
                    postingsList.clear();
                }
            }

            int sum = 0;// number of one word in cur file
            while(it.hasNext()) {
                sum += it.next().get();
            }
            total_sum += sum;
            prev = cur_term;
            postingsList.add(String.format("%s: %d;", cur_file, sum));
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            if(total_sum > threshold) {
                String out_k = String.format("[%s] %d,", prev, total_sum);
                OUT_KEY.set(out_k);

                String out_v = "";
                for(String s: postingsList) out_v += s;
                OUT_VALUE.set(out_v.substring(0, out_v.length() - 1));
                context.write(OUT_KEY, OUT_VALUE);
            }
        }
    }

    public static void main(String[] args) {// args[0]: reliable package, args[1]: input args[2]:output
        try {
            Configuration conf = new Configuration();

            DistributedCache.addCacheFile((new Path(args[1] + "cn_stopwords.txt")).toUri(), conf);
            Job job = new Job(conf, "tokenize");

            job.setJarByClass(tokenize.class);

            job.setMapperClass(TkMapper.class);
            job.setPartitionerClass(TkPartitioner.class);
            job.setReducerClass(TkReducer.class);

            job.setMapOutputKeyClass(Text.class);
            job.setMapOutputValueClass(IntWritable.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(Text.class);

            FileSystem fs = FileSystem.get(conf);

            FileStatus[] package_status = fs.listStatus(new Path(args[0]));
            for(FileStatus f: package_status) {
                job.addFileToClassPath(f.getPath());
            }

            FileStatus[] status = fs.listStatus(new Path(args[1] + "test"));// or train
            for(FileStatus f: status) { 
                System.out.println(f.getPath());
                if(f.isDir()) FileInputFormat.addInputPath(job, f.getPath());
            }
            FileOutputFormat.setOutputPath(job, new Path(args[2]));
            System.exit(job.waitForCompletion(true) ? 0 : 1);
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }
}