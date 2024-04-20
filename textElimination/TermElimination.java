import java.io.IOException;
import java.util.StringTokenizer;

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

public class TermElimination {

    public static class TermEliminationMapper extends Mapper<Object, Text, Text, IntWritable>{

        private final static IntWritable one = new IntWritable(1);
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
    public static class TermEliminationReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
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

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();
    Job job = Job.getInstance(conf, "Term Elimination");
    job.setJarByClass(TermElimination.class);
    job.setMapperClass(TermEliminationMapper.class);
    job.setCombinerClass(TermEliminationReducer.class);
    job.setReducerClass(TermEliminationReducer.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);

    Path outputPath = new Path(args[1]);
    FileSystem fs = FileSystem.get(conf);
    if (fs.exists(outputPath)) {
        fs.delete(outputPath, true);
    }

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
