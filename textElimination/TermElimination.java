import java.io.IOException;
import java.util.StringTokenizer;
import java.net.URI;
import java.util.List;
import java.util.ArrayList;
import java.io.BufferedReader;
import java.io.FileReader;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class TermElimination {

    public static class SuitableTermMapper extends Mapper<Object, Text, Text, IntWritable>{

        private Text termId = new Text();
        private IntWritable freq = new IntWritable();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        // value format: termId    docId    freq
            String[] parts = value.toString().split("\\s+");
            termId.set(parts[0]);
            freq.set(Integer.parseInt(parts[2]));
            context.write(termId, freq);
        }
        
    }
    public static class SuitableTermReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();

        public void reduce(Text key, Iterable<IntWritable> values,Context context) throws IOException, InterruptedException {
            int totalFreq = 0;

            for (IntWritable value : values) {
                totalFreq += value.get();
            }

            if (totalFreq >= 3) {
                result.set(totalFreq);
                context.write(key, result);
            }
        }
    }

    public static class TermEliminationMapper extends Mapper<Object, Text, Text, IntWritable> {

        private List<String> suitableTerms = new ArrayList<>();

        @Override
        // stored suitable terms in list
        public void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            Path filePath = new Path(cacheFiles[0].toString());
            String fileName = filePath.getName();
            BufferedReader reader = new BufferedReader(new FileReader(fileName));

            String line;
            while ((line = reader.readLine()) != null) {
                // line: term_id freq 
                String termId = line.split("\\s+")[0];
                suitableTerms.add(termId);
            }
            reader.close();
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            //input: termId   docId    freq
            String[] parts = value.toString().split("\\s+");
            String termId = parts[0];

            if (suitableTerms.contains(termId)) {
                context.write(new Text(parts[0] + " " + parts[1] + " " + parts[2]), new IntWritable(1));
            }

        }
    }

    public static class TermEliminationReducer extends Reducer<Text, IntWritable, Text, NullWritable> {

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            context.write(key, NullWritable.get());
        }
    }
    

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    String input = args[0];
    String output = args[1];


    FileSystem fs = FileSystem.get(conf);
    Path outputDir = new Path(output);
    if (fs.exists(outputDir)) {
        fs.delete(outputDir, true);
    }
    

     // job1: find suitable terms (freq >= 3)
    Job job1 = Job.getInstance(conf, "Suitable Terms");
    job1.setJarByClass(TermElimination.class);
    job1.setMapperClass(SuitableTermMapper.class);
    job1.setCombinerClass(SuitableTermReducer.class);
    job1.setReducerClass(SuitableTermReducer.class);
    job1.setOutputKeyClass(Text.class);
    job1.setOutputValueClass(IntWritable.class);

    String findSuitableTermsOutput = output + "/suitableTerms";
    FileInputFormat.addInputPath(job1, new Path(args[0]));
    FileOutputFormat.setOutputPath(job1, new Path(findSuitableTermsOutput));

    // stop when job1 failed
    if(!job1.waitForCompletion(true)) {
        System.exit(1);
    }

    Job job2 = Job.getInstance(conf, "Terms Elimination");
    job2.setJarByClass(TermElimination.class);
    job2.setMapperClass(TermEliminationMapper.class);
    job2.setReducerClass(TermEliminationReducer.class);
    job2.setOutputKeyClass(Text.class);
    job2.setOutputValueClass(IntWritable.class);

    String suitableTermsInput = findSuitableTermsOutput + "/part-r-00000";
    job2.addCacheFile(new URI(suitableTermsInput));
    FileInputFormat.addInputPath(job2, new Path(args[0]));
    FileOutputFormat.setOutputPath(job2, new Path(args[1] + "/eliminatedTerms"));

    System.exit(job2.waitForCompletion(true) ? 0 : 1);

  }
}
