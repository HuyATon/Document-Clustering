import java.io.IOException;
import java.util.StringTokenizer;
import java.util.Map;
import java.util.HashMap;
import java.util.HashSet;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Set;
import java.net.URI;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;
import org.apache.hadoop.fs.FileSystem;


public class TermFrequency {

    public static class TermFrequencyMapper extends Mapper<Object, Text, Text, IntWritable> {
        private static final Set<String> stopWords = new HashSet<>();
        private static final Map<String, Integer> termIdMap = new HashMap<>();
        private static final Map<String, Integer> docIdMap = new HashMap<>();
        Text word = new Text();
        private static final IntWritable one = new IntWritable(1);
        private IntWritable termId = new IntWritable();
        private IntWritable docId = new IntWritable();
        

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles();
            for (int i = 0; i < cacheFiles.length; i++) {
                String fileName = new Path(cacheFiles[i].toString()).getName();
                try(BufferedReader br = new BufferedReader(new FileReader(fileName))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (i == 0 ) {
                            stopWords.add(line.trim().toLowerCase());
                        }
                        else if (i == 1 ) {
                            int id = 1;
                            while((line = br.readLine()) != null) {
                                termIdMap.put(line.trim().toLowerCase(), id++);
                            }
                        }
                        else {
                            int id = 1;
                            while((line = br.readLine()) != null) {
                                docIdMap.put(line.trim().toLowerCase(), id++);
                            }
                        }
                    }
                }
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            Path filePath = ((FileSplit) context.getInputSplit()).getPath();
            String docName = filePath.getParent().getName() + "." + filePath.getName().replace(".txt", "");
            String line = value.toString().toLowerCase();
            StringTokenizer itr = new StringTokenizer(line, "\t\n\r\f,.:;?![]'\"()& ");

            if (docIdMap.get(docName) == null) {
                return;
            }
            docId.set(docIdMap.get(docName));
            while(itr.hasMoreTokens()) {
                String token = itr.nextToken();
                if (!stopWords.contains(token)) {
                    if (termIdMap.get(token) == null) {
                        continue;
                    }
                    termId.set(termIdMap.get(token));
                    word.set(termId + " " + docId);
                    context.write(word, one);
                }
            }
        }
    }
    public static class TermFrequencyReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        private IntWritable result = new IntWritable();
        public void reduce(Text key, Iterable<IntWritable> entries, Context context) throws IOException, InterruptedException {
           
           int sum =0;
           for (IntWritable entry: entries) {
               sum += entry.get();
           }
            result.set(sum);
            context.write(key, result);
        }

    }

    public static void main(String[] args) throws Exception {
        // <input file> <output file>
        Configuration conf = new Configuration();

        // auto remove output folder
        Path outputPath = new Path(args[1]);
        FileSystem fs = FileSystem.get(conf);
        if (fs.exists(outputPath)) {
            fs.delete(outputPath, true);
        }


        Job job = Job.getInstance(conf, "term frequency");
        job.setJarByClass(TermFrequency.class);
        job.setMapperClass(TermFrequencyMapper.class);
        job.setCombinerClass(TermFrequencyReducer.class);
        job.setReducerClass(TermFrequencyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        // add stopwords.txt, .terms, .docs as cache files (in input folders)
        // job.addCacheFile(new URI("/textCleaning/stopwords.txt"));
        // job.addCacheFile(new URI("/textCleaning/bbc.terms"));
        // job.addCacheFile(new URI("/textCleaning/bbc.docs"));

        fs.copyFromLocalFile(new Path("./stopwords.txt"), new Path("/stopwords.txt"));
        fs.copyFromLocalFile(new Path("./bbc.terms"), new Path("/bbc.terms"));
        fs.copyFromLocalFile(new Path("./bbc.docs"), new Path("/bbc.docs"));

        job.addCacheFile(new URI("/stopwords.txt"));
        job.addCacheFile(new URI("/bbc.terms"));
        job.addCacheFile(new URI("/bbc.docs"));
        
        // set for reading files in subdirectories
        FileInputFormat.setInputDirRecursive(job, true);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, outputPath);
        System.exit(job.waitForCompletion(true) ? 0 : 1);
}
}

