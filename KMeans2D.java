import java.io.*;
import java.util.*;
import java.net.URI;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.logging.Logger;


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

import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.fs.FSDataInputStream;



public class KMeans2D {

    public static class Point {
        double x;
        double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    public static double euclideanDistance(Point p1, Point p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    }

    public static Point average(List<Point> points) {
        double sumX = 0, sumY = 0;
        for (Point point : points) {
            sumX += point.x;
            sumY += point.y;
        }
        return new Point(sumX / points.size(), sumY / points.size());
    }

    public static class ClusteringMapper extends Mapper<Object, Text, Text, Text> {
        List<Point> centroids = new ArrayList<>();



        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            try {
                // Get the iteration number
                int iteration = context.getConfiguration().getInt("iteration", 0);
                // Determine the path of the centroids file from the previous iteration
                Path centroidsPath;
                if (iteration == 0) {
                    for (int i = 0; i < 3; i++) {
                        centroids.add(new Point(Math.random() * 10, Math.random() * 10));
                    }
                    
                } else {
                    centroidsPath = new Path("/output/centers_" + (iteration - 1) + "/centers-r-00000"); 
                    BufferedReader reader = new BufferedReader(new InputStreamReader(FileSystem.get(context.getConfiguration()).open(centroidsPath)));
                    String line;
                    while ((line = reader.readLine()) != null) {
                        String[] parts = line.split("\\s+");
                        double x = Double.parseDouble(parts[1]);
                        double y = Double.parseDouble(parts[2]);
                        centroids.add(new Point(x, y));
                    }
                    reader.close();
                }
                
            } catch (Exception e) {
                e.printStackTrace();

            }
        }
        
        

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] parts = value.toString().split("\\s+");
            double x = Double.parseDouble(parts[0]);
            double y = Double.parseDouble(parts[1]);
            Point point = new Point(x, y);

            int minIndex = 0;
            double minDistance = 100000000;

            // Find the nearest centroid
            for (int i = 0; i < centroids.size(); i++) {
                double distance = euclideanDistance(point, centroids.get(i));
                if (distance < minDistance) {
                    minDistance = distance;
                    minIndex = i;
                }
            }

            // Emit the point with the nearest centroid's index
            context.write(new Text(Integer.toString(minIndex)), value);
        }
    }


    public static class ClusteringReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text value : values) {
                context.write(key, value); 
            }
        }
    }
    

    public static class UpdateCentroidsMapper extends Mapper<Object, Text, Text, Text> {
        private URI[] cacheFiles;
    
        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            cacheFiles = context.getCacheFiles();
        }
    
        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            if (cacheFiles != null && cacheFiles.length > 0) {
                FileSystem fs = FileSystem.get(context.getConfiguration());
                Path path = new Path(cacheFiles[0]);
                FSDataInputStream fsDataInputStream = fs.open(path);
                BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fsDataInputStream));
                String line;
                while ((line = bufferedReader.readLine()) != null) {
                    String[] tokens = line.split("\\s+");
                    if (tokens.length == 3) {
                        context.write(new Text(tokens[0]), new Text(tokens[1] + "," + tokens[2]));
                    }
                }
    
                bufferedReader.close();
                fsDataInputStream.close();
            }
        }
    }
    
    public static class UpdateCentroidsReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            double sumX = 0.0;
            double sumY = 0.0;
            int count = 0;
    
            for (Text value : values) {
                String[] coords = value.toString().split(",");
                sumX += Double.parseDouble(coords[0]);
                sumY += Double.parseDouble(coords[1]);
                count++;
            }
    
            double newX = sumX / count;
            double newY = sumY / count;
            context.write(key, new Text(newX + " " + newY));
        }
    }
    
    

    public static void main(String[] args) throws Exception {

        int iteration = 0;
        int numIterations = 20; // Set the number of iterations as required

        
        String input = args[0];  
        String output = args[1]; 
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);

        // Delete output directory if it exists
        if (fs.exists(new Path(output))) {
            fs.delete(new Path(output), true);
        }

        while (iteration < numIterations) {
            // Job: CLUSTERING
            Job clusteringJob = Job.getInstance(conf, "clustering");
            // Before running the clustering job, set the iteration number
            clusteringJob.getConfiguration().setInt("iteration", iteration);

            clusteringJob.setJarByClass(KMeans2D.class);
            clusteringJob.setMapperClass(ClusteringMapper.class);
            clusteringJob.setReducerClass(ClusteringReducer.class);
            clusteringJob.setOutputKeyClass(Text.class);
            clusteringJob.setOutputValueClass(Text.class);

            String clustering_output = output + "/classes_" + String.valueOf(iteration);
            FileInputFormat.addInputPath(clusteringJob, new Path(input));
            FileOutputFormat.setOutputPath(clusteringJob, new Path(clustering_output));
            clusteringJob.getConfiguration().set("mapreduce.output.basename", "classes");
            clusteringJob.waitForCompletion(true);

            // Job: UPDATE CENTROIDS

            Job updateCentroidsJob = Job.getInstance(new Configuration(), "update centroids");

            updateCentroidsJob.setJarByClass(KMeans2D.class);
            updateCentroidsJob.setMapperClass(UpdateCentroidsMapper.class);
            updateCentroidsJob.setReducerClass(UpdateCentroidsReducer.class);
            updateCentroidsJob.setOutputKeyClass(Text.class);
            updateCentroidsJob.setOutputValueClass(Text.class);

            String updateCentroidsOutput = output + "/centers_" + String.valueOf(iteration);
            updateCentroidsJob.addCacheFile(new URI(clustering_output + "/classes-r-00000"));


            updateCentroidsJob.setInputFormatClass(TextInputFormat.class); 
            FileInputFormat.addInputPath(updateCentroidsJob, new Path(input));
            FileOutputFormat.setOutputPath(updateCentroidsJob, new Path(updateCentroidsOutput));
            updateCentroidsJob.getConfiguration().set("mapreduce.output.basename", "centers");
            updateCentroidsJob.waitForCompletion(true);

            iteration++;
        }


        // Writing clusters and classes data for the last iteration
        int lastIteration = numIterations - 1;
        Path oldClustersFilePath = new Path(output + "/centers_" + lastIteration + "/centers-r-00000");
        Path oldClassesFilePath = new Path(output + "/classes_" + lastIteration + "/classes-r-00000");
        Path newClustersFilePath = new Path(output + "/task_2_1.clusters");
        Path newClassesFilePath = new Path(output + "/task_2_1.classes");

        if (fs.exists(oldClustersFilePath)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(oldClustersFilePath)));
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(newClustersFilePath)));
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
            reader.close();
            writer.close();
        } else {
            System.err.println("Centers file not found for the last iteration");
        }

        if (fs.exists(oldClassesFilePath)) {
            BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(oldClassesFilePath)));
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(fs.create(newClassesFilePath)));
            String line;
            while ((line = reader.readLine()) != null) {
                writer.write(line);
                writer.newLine();
            }
            reader.close();
            writer.close();
        } else {
            System.err.println("Classes file not found for the last iteration");
        }


    }
}
