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

    
    public static class FrequencyCountMapper extends Mapper<Object, Text, Text, IntWritable>{
        
        Text termId = new Text();
        IntWritable freq = new IntWritable();

        // term_id   doc_id    freq
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            termId.set(parts[0]);
            freq.set(Integer.parseInt(parts[2]));
            context.write(termId, freq);
        }
    }

    public static class FrequencyCountReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        Text termId = new Text();
        IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,Context context) throws IOException, InterruptedException {
            int totalFreq = 0;
            
            for (IntWritable value : values) {
                totalFreq += value.get();
            }
            result.set(totalFreq);
            context.write(key, result);
        }
    }


    // MAPPER
    public static class TopTermMapper extends Mapper<Object, Text, Text, IntWritable>{
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

    Path[] outputPaths = {new Path(args[1]), new Path(args[2])};
    FileSystem fs = FileSystem.get(conf);
    for(Path outputPath : outputPaths) {
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }
    }


    // usage: <input job1> <output job1> <output job2>
    // JOB1: COUNT FREQUENCY
    Job job1 = Job.getInstance(conf, "Frequency Count");
    job1.setJarByClass(TopTerm.class);
    job1.setMapperClass(FrequencyCountMapper.class);
    job1.setCombinerClass(FrequencyCountReducer.class);
    job1.setReducerClass(FrequencyCountReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(IntWritable.class);


    FileInputFormat.addInputPath(job1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job1, new Path(args[1]));
    
    if (!job1.waitForCompletion(true)) {
        System.exit(1);
    }

    // JOB 2: Top 10 Terms
    Job job2 = Job.getInstance(conf, "Top 10 Terms");
    job2.setJarByClass(TopTerm.class);
    job2.setMapperClass(TopTermMapper.class);
    job2.setReducerClass(TopTermReducer.class);
    job2.setOutputKeyClass(Text.class);
    job2.setOutputValueClass(IntWritable.class);

    
    // make sure only one reducer is used
    job2.setNumReduceTasks(1);

    FileInputFormat.addInputPath(job2, new Path(args[1]));
    FileOutputFormat.setOutputPath(job2, new Path(args[2]));
    System.exit(job2.waitForCompletion(true) ? 0 : 1);
  }
}