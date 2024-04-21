import java.util.*;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.net.URI;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

public class TF_IDF {

    // job1: calculate idf(term):

    public static class IDFMapper extends Mapper<Object, Text, Text, DoubleWritable> {
        Text termId = new Text();
        DoubleWritable one = new DoubleWritable(1);
        

        // input: term_id   doc_id    freq
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            termId.set(parts[0]);
            context.write(termId, one);
        }
    }
    public static class IDFReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        Text termId = new Text();
        DoubleWritable idf = new DoubleWritable();
        // input: term_id  [1, 1, 1, ...] (count the document in which term_id exist)
        // output: term_id   idf_value
        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
         
            // count the number of documents in which term_id exist
            double D = 50; // because we have 50 documents
            double count = 0;

            for (DoubleWritable val : values) {
                count += val.get();
            }
            termId.set(key);
            idf.set(Math.log((double)D / count));
            context.write(termId, idf);
        }
    }

    // job2: count #words in each document
    public static class WordCountMapper extends Mapper<Object, Text, Text, IntWritable> {
        // input: term_id   doc_id    freq
        Text docId = new Text();
        IntWritable term_freq = new IntWritable();
        

        // output: doc_id, count_of_current_term
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            docId.set(parts[1]);
            term_freq.set(Integer.parseInt(parts[2]));
            context.write(docId, term_freq);
        }
    }
    public static class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        Text docId = new Text();
        IntWritable result = new IntWritable();

        // input: doc_id  [freq1, freq2, freq3, ...]
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            // output: doc_id   #words
            context.write(key, result);
        }
    }
    
    // job3: calculate tf-idf(term, doc)

    public static class TFIDF_Mapper extends Mapper<Object, Text, Text, DoubleWritable> {
        // input: term_id   doc_id    freq
        Text termId = new Text();
        Text docId = new Text();
        DoubleWritable tf_idf = new DoubleWritable();
        Map<String, Integer> documentLength = new HashMap<>();
        Map<String, Double> idfMap = new HashMap<>();
        
        // create a HashMap store: <docId, #words>
        // create a HashMap store: <docId, idf_value>
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            Path documentLengthPath = new Path(context.getCacheFiles()[0].toString());
            BufferedReader reader = new BufferedReader(new FileReader(documentLengthPath.getName()));
            String line;

            while ((line = reader.readLine()) != null) {
                // line: docid    #words
                String[] parts = line.split("\\s+");
                String docId = parts[0];
                String words = parts[1];
                documentLength.put(docId, Integer.parseInt(words));
            }
            reader.close();

            Path idfPath = new Path(context.getCacheFiles()[1].toString());
            BufferedReader reader2 = new BufferedReader(new FileReader(idfPath.getName()));
            String line2;
            while((line2 = reader2.readLine()) != null) {
                // line: term_id    idf_value
                String[] parts = line2.split("\\s+");
                String termId = parts[0];
                Double idf = Double.parseDouble(parts[1]);
                idfMap.put(termId, idf);
            }
            reader2.close();
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            // input: term_id    doc_id      freq
            String[] parts = value.toString().split("\\s+");
            String currentTermId = parts[0];
            String currentDocId = parts[1];
            Integer currentFreq = Integer.parseInt(parts[2]);
            Integer words = documentLength.get(currentDocId);

            termId.set(currentTermId);
            docId.set(currentDocId);
            double tf = (double)currentFreq / words;
            double idf = idfMap.get(currentTermId);
            tf_idf.set(tf * idf);

            // output: term_id      doc_id      tf_values
            context.write(new Text(termId.toString() + " " + docId.toString()), tf_idf);
        }
    }
    public static class TFIDF_Reducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {
        DoubleWritable result = new DoubleWritable();

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {

            // input: "term_id      doc_id", [tf1, tf2...] (but actually we just have 1 tf)
            double sum = 0;

            for (DoubleWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);
        }
    }


    public static void main(String[] args) throws Exception {
        // <input file> <output file>
        Configuration conf = new Configuration();


        Path idfInput = new Path(args[0]);
        Path idfOutput = new Path(args[0].split("input")[0] + "job1/output");

        Path docLengthInput = new Path(args[0]);
        Path docLengthOutput = new Path(args[0].split("input")[0] + "job2/output");

        Path tf_idfInput = new Path(args[0]);
        Path tf_idfOutput = new Path(args[1]);

        // auto remove output folder
        Path outputPath = new Path(args[1]);
        Path[] paths = {idfOutput, docLengthOutput, tf_idfOutput};
        FileSystem fs = FileSystem.get(conf);
        for (Path path: paths) {
            if (fs.exists(path)) {
                fs.delete(path, true);
            }
        }
        // job 1: calculate idf(term)
        Job job1 = Job.getInstance(conf, "calculate idf");
        job1.setJarByClass(TF_IDF.class);
        job1.setMapperClass(IDFMapper.class);
        job1.setReducerClass(IDFReducer.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(DoubleWritable.class);

        FileInputFormat.addInputPath(job1, idfInput); 
        FileOutputFormat.setOutputPath(job1, idfOutput);
        
        if (!job1.waitForCompletion(true)) {
            System.out.print("JOB 1 FAILED");
            System.exit(1);
        }

        // job 2: count #words in each document
        Job job2 = Job.getInstance(conf, "calculate document length");
        job2.setJarByClass(TF_IDF.class);
        job2.setMapperClass(WordCountMapper.class);
        job2.setReducerClass(WordCountReducer.class);
        job2.setOutputKeyClass(Text.class);
        job2.setOutputValueClass(IntWritable.class);
        
        FileInputFormat.addInputPath(job2, docLengthInput);   // file task_1_2.mtx (term   doc   freq)
        FileOutputFormat.setOutputPath(job2, docLengthOutput); // output: doc_id    #words

        if (!job2.waitForCompletion(true)) {
            System.out.print("JOB 2 FAILED");
            System.exit(1);
        }

        // job 3: calculate tf-idf(term, doc)
        Job job3 = Job.getInstance(conf, "calculate tf-idf");
        job3.setJarByClass(TF_IDF.class);
        job3.setMapperClass(TFIDF_Mapper.class);
        job3.setReducerClass(TFIDF_Reducer.class);
        job3.setOutputKeyClass(Text.class);
        job3.setOutputValueClass(DoubleWritable.class);

        Path idfOutputHDFS = new Path(idfOutput + "/part-r-00000");
        Path docLengthOutputHDFS = new Path(docLengthOutput + "/part-r-00000");
        Path idfOutputLocal = new Path("./idfOutput.txt");
        Path docLengthOutputLocal = new Path("./docLengthOutput.txt");

        fs.copyToLocalFile(idfOutputHDFS, idfOutputLocal);
        fs.copyToLocalFile(docLengthOutputHDFS, docLengthOutputLocal);
        fs.copyFromLocalFile(new Path("./idfOutput.txt"), new Path("/idfOutput.txt"));
        fs.copyFromLocalFile(new Path("./docLengthOutput.txt"), new Path("/docLengthOutput.txt"));

        job3.addCacheFile(new URI("/docLengthOutput.txt"));
        job3.addCacheFile(new URI("/idfOutput.txt"));

        
        FileInputFormat.setInputDirRecursive(job3, true);
        FileInputFormat.addInputPath(job3, tf_idfInput);
        FileOutputFormat.setOutputPath(job3, tf_idfOutput);
        System.exit(job3.waitForCompletion(true) ? 0 : 1);

        
    }
    
}