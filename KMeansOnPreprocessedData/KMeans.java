import java.io.*;
import java.util.*;
import java.net.URI;
import java.io.BufferedReader;
import java.io.FileReader;




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
    public static List<Integer> parseTermId(String s) {
        List<Integer> termIds = new ArrayList<>();
        String[] termId_tfidf = s.split(",");
        for (String pair : termId_tfidf) {
            String[] parts = pair.split(":");
            termIds.add(Integer.parseInt(parts[0]));
        }
        return termIds;
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
                    for (int j = 0; j < 464; j++) { // 464 terms in total
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
            List<List<Double>> tfidfOfPoints = new ArrayList<>();
            List<Integer> termIds = new ArrayList<>();
            for (Text value: values) {
                String value_string = value.toString();
                tfidfOfPoints.add(parseTFIDF(value_string));
                updatedCentroid = average(tfidfOfPoints);

                if (termIds.isEmpty()) {
                    termIds = parseTermId(value_string);
                }
            }          
            String formattedString = "";
            for (int i = 0; i < updatedCentroid.size(); i++) {
                formattedString += termIds.get(i) + ":" + updatedCentroid.get(i) + ",";
            }
            formattedString = formattedString.substring(0, formattedString.length() - 1); // remove last comma
            context.write(key, new Text(formattedString));

        }
      }

      // job:  (LOSS)
      public static class LossMapper extends  Mapper<Object, Text, Text, DoubleWritable> {
        Map<Integer, List<Double>> centroids = new HashMap<>();
        Map<Integer, Integer> docId_centroids = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            URI[] cacheFiles = context.getCacheFiles(); 
            File file = new File("centers");
            
            for (URI cacheFile : cacheFiles ) {
                Path filePath = new Path(cacheFile);
                String fileName = filePath.getName();
                String line;
                if (fileName.contains("centers")) {
                    BufferedReader reader = new BufferedReader(new FileReader(fileName));
        
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\\s+");
                        Integer clusterId = Integer.parseInt(parts[0]);
                        List<Double> tfidf = parseTFIDF(parts[1]);
                        centroids.put(clusterId, tfidf);
                    }
                }
                else  {
                    BufferedReader reader = new BufferedReader(new FileReader(fileName));
        
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\\s+");
                        Integer docId = Integer.parseInt(parts[0]);
                        Integer clusterId = Integer.parseInt(parts[1]);
                        docId_centroids.put(docId, clusterId);
                    }
                }
            }
        }

      public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] parts = value.toString().split("\\|");
        Integer docId = Integer.parseInt(parts[0]);
        Integer clusterId = docId_centroids.get(docId);
        List<Double> centroid_tfdif = centroids.get(clusterId);
        List<Double> tfidf = parseTFIDF(parts[1]);
        double dist = 1 - cosineSimilarity(centroid_tfdif, tfidf);
        context.write(new Text("Loss"), new DoubleWritable(dist));
      }
    }

    public static class LossReducer extends Reducer<Text, DoubleWritable, Text, DoubleWritable> {

        public void reduce(Text key, Iterable<DoubleWritable> values, Context context) throws IOException, InterruptedException {
            double loss = 0;
            for (DoubleWritable value : values) {
                loss += value.get();
            }
            context.write(key, new DoubleWritable(loss));
        }
    }
      // job: TOP 10 words
      public static class Top10Mapper extends Mapper<Object, Text, Text, Text> {

        TreeMap<Double, String> top10 = new TreeMap<>(); // <termId, tfidf>
        Text clusterId = new Text();

        //  input: centroids file

        // format: centroids (\tab) tfidfs
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            // output of map: <clusterId, top 10 terms>
            String[] parts = value.toString().split("\\s+");
            clusterId.set(parts[0].toString());
            List<Integer> termIds = parseTermId(parts[1]);
            List<Double> tfidfs = parseTFIDF(parts[1]);

            // get 10 words with highest tfidf
            for (int i = 0; i < tfidfs.size(); i++) {
                top10.put(tfidfs.get(i), termIds.get(i).toString());
                if (top10.size() > 10) {
                    top10.remove(top10.firstKey());
                }
            }
            String top10ByDescending = "";

            for (Map.Entry<Double, String> entry : top10.descendingMap().entrySet() ) {
                String termId = entry.getValue();
                top10ByDescending += termId + ",";
            }
            context.write(clusterId, new Text(top10ByDescending));
        }

    }

    public static class Top10Reducer extends Reducer<Text, Text, Text, Text> {

        // format output: clusterId,  word_top
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                // just 1 iteration actually
                context.write(key, value);
            }
        }
    }

  
  

  public static void main(String[] args) throws Exception {
   
    int iteration = 0;
    int numIterations = 2;
    String input = args[0];
    String output = args[1];
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(new Configuration());

    if (fs.exists(new Path(output))) {
        fs.delete(new Path(output), true);
    }

    while (iteration < numIterations) {

        // job: CLUSTERING
        Job clusteringJob = Job.getInstance(new Configuration(), "clustering");
        clusteringJob.setJarByClass(KMeans.class);
        clusteringJob.setMapperClass(ClusteringMapper.class);
        clusteringJob.setReducerClass(ClusteringReducer.class);
        clusteringJob.setOutputKeyClass(Text.class);
        clusteringJob.setOutputValueClass(IntWritable.class);


        String clustering_output = output + "/classes_" + String.valueOf(iteration);
        FileInputFormat.addInputPath(clusteringJob, new Path(input));
        FileOutputFormat.setOutputPath(clusteringJob, new Path(clustering_output));
        clusteringJob.getConfiguration().set("mapreduce.output.basename", "classes");
        clusteringJob.waitForCompletion(true);

        // job: UPDATE CENTROIDS
        Job updateCentroidsJob = Job.getInstance(new Configuration(), "update centroids");
        updateCentroidsJob.setJarByClass(KMeans.class);
        updateCentroidsJob.setMapperClass(UpdateCentroidsMapper.class);
        updateCentroidsJob.setReducerClass(UpdateCentroidsReducer.class);
        updateCentroidsJob.setOutputKeyClass(Text.class);
        updateCentroidsJob.setOutputValueClass(Text.class);

        String updateCentroidsOutput = output + "/centers_" + String.valueOf(iteration);
        updateCentroidsJob.addCacheFile(new URI(clustering_output + "/classes-r-00000"));
        FileInputFormat.addInputPath(updateCentroidsJob, new Path(input));
        FileOutputFormat.setOutputPath(updateCentroidsJob, new Path(updateCentroidsOutput));
        updateCentroidsJob.getConfiguration().set("mapreduce.output.basename", "centers");
        updateCentroidsJob.waitForCompletion(true);


        

        // job: COMPUTE LOSS
        Job lossJob = Job.getInstance(new Configuration(), "compute loss");
        lossJob.setJarByClass(KMeans.class);
        lossJob.setMapperClass(LossMapper.class);
        lossJob.setReducerClass(LossReducer.class);
        lossJob.setOutputKeyClass(Text.class);
        lossJob.setOutputValueClass(DoubleWritable.class);

        String lossOutput = output + "/loss_" + String.valueOf(iteration);
        
        lossJob.addCacheFile(new URI(updateCentroidsOutput + "/centers-r-00000"));
        lossJob.addCacheFile(new URI(clustering_output + "/classes-r-00000"));

        FileInputFormat.addInputPath(lossJob, new Path(input));
        FileOutputFormat.setOutputPath(lossJob, new Path(lossOutput));
        lossJob.getConfiguration().set("mapreduce.output.basename", "loss");
        lossJob.waitForCompletion(true);



        // job: TOP 10 WORDS
        Job top10Job = Job.getInstance(new Configuration(), "top 10 words");
        top10Job.setJarByClass(KMeans.class);
        top10Job.setMapperClass(Top10Mapper.class);
        top10Job.setReducerClass(Top10Reducer.class);
        top10Job.setOutputKeyClass(Text.class);
        top10Job.setOutputValueClass(Text.class);

        String top10WordsOutput = output + "/top10_" + String.valueOf(iteration);

        FileInputFormat.addInputPath(top10Job, new Path(updateCentroidsOutput + "/centers-r-00000"));
        FileOutputFormat.setOutputPath(top10Job, new Path(top10WordsOutput));
        top10Job.getConfiguration().set("mapreduce.output.basename", "top10");
        top10Job.waitForCompletion(true);




        iteration++;

    }
  }
}
