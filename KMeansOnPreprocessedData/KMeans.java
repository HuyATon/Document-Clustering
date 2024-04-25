import java.io.*;
import java.util.*;
import java.net.URI;
import java.io.BufferedReader;
import java.io.FileReader;




import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.xbill.DNS.tools.update;
import org.apache.hadoop.fs.FileSystem;





public class KMeans {

    public static List<Double> parseTFIDF(String s) {
        
        List<Double> values = new ArrayList<>();
        String[] termId_tfidf = s.split(",");
        for (String pair : termId_tfidf) {
            String[] parts = pair.split(":");
            values.add(Double.parseDouble(parts[1]));
        }
        return values;
    }

    public static double cosineSimilarity(List<Double> centroid, List<Double> point) {
        double dotProduct = 0;
        double normA = 0;
        double normB = 0;
        for (int i = 0; i < centroid.size(); i++) {
            dotProduct += centroid.get(i) * point.get(i);
            normA += Math.pow(centroid.get(i), 2);
            normB += Math.pow(point.get(i), 2);
        }
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    
    public static List<Double> average(List<List<Double>> points) {
        List<Double> avg = new ArrayList<>();
        for (int i = 0; i < points.get(0).size(); i++) {
            double sum = 0;
            for (List<Double> point : points) {
                sum += point.get(i);
            }
            avg.add(sum / points.size());
        }
        return avg;
    }

    public static class ClusteringMapper extends Mapper<Object, Text, Text, IntWritable> {
        Map<Integer, List<Double>> centroids = new HashMap<>();


        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            try {
                URI centroidFile = context.getCacheFiles()[0];
                Path filePath = new Path(centroidFile.toString());
                String fileName = filePath.getName();

                BufferedReader reader = new BufferedReader(new FileReader(fileName));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\\s+");
                        int clusterId = Integer.parseInt(parts[0]);
                        List<Double> values = parseTFIDF(parts[1]);
                    }
            }
            catch (Exception e) {
                // randome centroids
                for (int i = 0; i < 5; i++) {
                    List<Double> values = new ArrayList<>();
                    for (int j = 0; j < 464; j++) {
                        values.add(Math.random());
                    }
                    centroids.put(i, values);
                }
            }
            
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
           
            String[] parts = value.toString().split("\\|");
            Text docId = new Text();
            IntWritable cluster = new IntWritable();
            
            docId.set(parts[0]);
            List<Double> point = parseTFIDF(parts[1]);

            double maxScore = 0;
            for (int i = 0; i < 5; i++) {
                double score = cosineSimilarity(centroids.get(i), point);
                if (score > maxScore) {
                    maxScore = score;
                    cluster.set(i);
                }
            }
            context.write(docId, cluster);
        }
    }
    public static class ClusteringReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            IntWritable cluster = new IntWritable();
          
            for (IntWritable value : values) {
                cluster.set(value.get());
            }
            context.write(key, cluster);
        }
    }

    public static class UpdateCentroidsMapper extends Mapper<Object, Text, Text, Text> {
        Map<Integer, Integer> docId_centroids = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            // <docId, centroids>
            URI clusterResultFile = context.getCacheFiles()[0];
            Path filePath = new Path(clusterResultFile.toString());
            String fileName = filePath.getName();
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("\\s+");
                docId_centroids.put(Integer.parseInt(parts[0]), Integer.parseInt(parts[1]));
            }
        }

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        

            String[] parts = value.toString().split("\\|");
            Text values = new Text();
            Integer clusterId = docId_centroids.get(Integer.parseInt(parts[0]));
            Text cluster = new Text(clusterId.toString());
            values.set(parts[1]);
            context.write(cluster, values);
        }
    }

    public static class UpdateCentroidsReducer extends Reducer<Text,Text,Text,Text> {
        
    
        public void reduce(Text key, Iterable<Text> values,
                           Context context
                           ) throws IOException, InterruptedException {
          
            List<Double> updatedCentroid = new ArrayList<>();
            List<List<Double>> points = new ArrayList<>();
            for (Text value: values) {
                String value_string = value.toString();
                points.add(parseTFIDF(value_string));
                updatedCentroid = average(points);
            }          
            String foramttedString = "";
            for (int i = 0; i < updatedCentroid.size(); i++) {
                foramttedString += i + ":" + updatedCentroid.get(i) + ",";
            }
            context.write(key, new Text(foramttedString));

        }
      }
  
  

  public static void main(String[] args) throws Exception {
    // Configuration conf = new Configuration();
    // Job job = Job.getInstance(conf, "word count");
    // job.setJarByClass(WordCount.class);
    // job.setMapperClass(TokenizerMapper.class);
    // job.setCombinerClass(IntSumReducer.class);
    // job.setReducerClass(IntSumReducer.class);
    // job.setOutputKeyClass(Text.class);
    // job.setOutputValueClass(IntWritable.class);
    // FileInputFormat.addInputPath(job, new Path(args[0]));
    // FileOutputFormat.setOutputPath(job, new Path(args[1]));
    // System.exit(job.waitForCompletion(true) ? 0 : 1);
    int iteration = 0;
    int numIterations = 10;
    String input = args[0];
    String output = args[1];
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(new Configuration());

    if (fs.exists(new Path(output))) {
        fs.delete(new Path(output), true);
    }

    while (iteration < numIterations) {
        Job clusteringJob = Job.getInstance(new Configuration(), "clustering");
        clusteringJob.setJarByClass(KMeans.class);
        clusteringJob.setMapperClass(ClusteringMapper.class);
        clusteringJob.setReducerClass(ClusteringReducer.class);
        clusteringJob.setOutputKeyClass(Text.class);
        clusteringJob.setOutputValueClass(IntWritable.class);


        String clustering_output = output + "/clustering_" + String.valueOf(iteration);
        FileInputFormat.addInputPath(clusteringJob, new Path(input));
        FileOutputFormat.setOutputPath(clusteringJob, new Path(clustering_output));
        clusteringJob.waitForCompletion(true);

        Job updateCentroidsJob = Job.getInstance(new Configuration(), "update centroids");
        updateCentroidsJob.setJarByClass(KMeans.class);
        updateCentroidsJob.setMapperClass(UpdateCentroidsMapper.class);
        updateCentroidsJob.setReducerClass(UpdateCentroidsReducer.class);
        updateCentroidsJob.setOutputKeyClass(Text.class);
        updateCentroidsJob.setOutputValueClass(Text.class);

        String updateCentroidsOutput = output + "/centroids_" + String.valueOf(iteration);
        updateCentroidsJob.addCacheFile(new URI(clustering_output + "/part-r-00000"));
        FileInputFormat.addInputPath(updateCentroidsJob, new Path(input));
        FileOutputFormat.setOutputPath(updateCentroidsJob, new Path(updateCentroidsOutput));
        updateCentroidsJob.waitForCompletion(true);
        
        iteration++;
    }
  }
}
