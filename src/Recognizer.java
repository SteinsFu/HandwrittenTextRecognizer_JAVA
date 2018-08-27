import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Files;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

public class Recognizer {

    public static DecimalFormat f = new DecimalFormat("000");

    private double[][][] inputsSet;         //[row][col][imageCode]
    private int sampleSize;
    private Network network;

    public Recognizer(int sampleSize) {
        this.sampleSize = sampleSize;
        inputsSet = allSample();            // 20 set of "0123456789" images

        double[] input = new double[700];   //25x28 = 700 pixels
        double[] target = new double[10];   //10 targets for prob of 0123456789
        int[] layerSize = new int[] {input.length, 15, target.length};

        //create a new network:

        //network = new Network(input, target, layerSize);


        //import a old network:

        network = new Network("network3.txt");
        network.setTarget(new double[10]);      //avoid having null target

    }

    public void train(int iteration) {
        for (int iter = 0; iter < iteration; iter++) {
            //randomly train => generalization
            for (int samples = 0; samples < 10 * sampleSize; samples++) {
                int row = new Random().nextInt(10);
                int col = new Random().nextInt(sampleSize);

                double[] target = new double[10];       // 10 targets for prob of 0123456789
                target[row] = 1;

                try {
                    network.acceptSample(inputsSet[row][col], target);
                    network.train(1);
                } catch (Exception e) {
                    System.out.println("Fail to accept sample image");
                    e.printStackTrace();
                }

            }
        }
    }

    public void predict(String imageName) {
        network.predict(imageToArray(imageName));

        //guess the answer:
        double[] output = network.getOutput();
        int mostProb = 0;       //most probable prediction index [0,1,2,3,4,5,6,7,8,9]

        for (int i = 0; i < 10; i++) {
            if (output[i] > output[mostProb])
                mostProb = i;
        }

        System.out.println("Prediction : " + mostProb);

    }

    private double[][][] allSample() {             // 0 1 2 3 4 5 6 7 8 9 = one sample
        double[][][] sample = new double[10][sampleSize][];     //[row][col][imageCode]

        int imgIndex = 1;
        for (int row = 0; row < 10; row++) {                    //10 image for one col
            for (int col = 0; col < sampleSize; col++) {
                String imgName = "sample/image_part_" + f.format(imgIndex) + ".jpg";
                System.out.println(imgName + ":");
                sample[row][col] = imageToArray(imgName);

                //image size : 25x28
                /*
                int k = 0;
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 25; j++) {
                        System.out.print(f.format(sample[row][col][k]));
                        k++;
                    }
                    System.out.println();
                }
                */

                imgIndex++;
            }
        }

        return sample;
    }

    private static double[] imageToArray(String ImageName) {
        //open image
        File imgPath = new File(ImageName);
        BufferedImage img;
        double[] array = null;
        try {
           img = ImageIO.read(imgPath);
           BufferedImage grayscaleImage =
                   new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_ARGB);
           array = new double[img.getHeight() * img.getWidth()];
           //System.out.println(img.getWidth());
           //System.out.println(img.getHeight());
           int k = 0;
           for (int i = 0; i < img.getHeight(); i++) {
               for (int j = 0; j < img.getWidth(); j++) {
                   //get RGB on each pixel
                   Color c = new Color(img.getRGB(j, i));
                   int r = c.getRed();
                   int g = c.getGreen();
                   int b = c.getBlue();
                   int a = c.getAlpha();

                   int gray = (r + g + b) / 3;

                   array[k] = 1 - (gray / 255.0);           //convert to (white) 0.0 ~ 1.0 (black)
                   k++;
                   System.out.print(f.format(gray));
               }
               System.out.println();
           }

        } catch(IOException e) {
            e.printStackTrace();
        }

        return array;
    }
}
