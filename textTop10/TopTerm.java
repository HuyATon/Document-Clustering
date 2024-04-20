import java.io.IOException;
import java.util.StringTokenizer;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.TreeMap;
import java.util.Random;



import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class TopTerm {
    // MAPPER
    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable>{
        private TreeMap<Double, String> topMap;
        
        @Override
        public void setup(Context context) {
            topMap = new TreeMap<Double, String>();
        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            String termId = parts[0];
            Double freq = Double.parseDouble(parts[1]) + (new Random()).nextDouble();

            topMap.put(freq, termId);
            if (topMap.size() > 10) {
                topMap.remove(topMap.firstKey());
            }
        }
        @Override
        public void cleanup(Context context) throws IOException, InterruptedException{

            for (Map.Entry<Double, String> entry: topMap.entrySet()) {
                Text termId = new Text(entry.getValue());
                IntWritable freq = new IntWritable((int) Math.floor(entry.getKey()));

                context.write(termId, freq);
            }
        }
    }

    // REDUCER
    public static class TopTermReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        private TreeMap<Double, String> topMap;

        @Override
        public void setup(Context context) {
            topMap = new TreeMap<Double, String>();
        }

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            
            int freq = 0;
            String termId = key.toString();
            for (IntWritable val : values) {
                freq += val.get();
            }
            
            topMap.put((double) freq + (new Random()).nextDouble(), termId);

            if (topMap.size() > 10) {
                topMap.remove(topMap.firstKey());
            }
        }
        @Override
        public void cleanup(Context context) throws IOException, InterruptedException{

           for (Map.Entry<Double, String> entry: topMap.descendingMap().entrySet() ) {
                Text termId = new Text(entry.getValue());
                IntWritable freq = new IntWritable((int) Math.floor(entry.getKey()));
                context.write(termId, freq);
           }
        }
}

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Top 10 Terms");
    job.setJarByClass(TopTerm.class);
    job.setMapperClass(TokenizerMapper.class);
    job.setReducerClass(TopTermReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    Path outputPath = new Path(args[1]);
    FileSystem fs = FileSystem.get(conf);
    if (fs.exists(outputPath)) {
        fs.delete(outputPath, true);
    }
    // make sure only one reducer is used
    job.setNumReduceTasks(1);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}