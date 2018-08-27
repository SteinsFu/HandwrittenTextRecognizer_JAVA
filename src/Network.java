import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class Network {

    private int networkSize;
    private int[] layerSize;

    private Neuron[][] neuron;
    private double[] input;
    private double[] target;
    private double learningRate = 0.01;      //ideal: ~0.01
    private double[][] bias;                //[layer][neuron]

    private double[][] sampleInput;
    private double[][] sampleTarget;

    public Network(double[] input, double[] target, int[] layerSize) {
        this.networkSize = layerSize.length;
        this.layerSize = layerSize;
        this.input = input;
        this.target = target;
        this.bias = new double[networkSize][];
        //initialize bias with {0, 0, ...}
        for (int layer = 0; layer < networkSize; layer++) {
            bias[layer] = new double[layerSize[layer]];
        }

        System.out.println("##Network initializing...");

        iniNeurons(null);
    }

    //Import Network
    public Network(String filename) {
        try {
            Scanner scanner = new Scanner(new File(filename));
            //check if the file is blank
            if (!scanner.hasNext()) throw new IOException();

            //Reading...
            //new fields
            String[] newLayerSizeString = scanner.nextLine().split(" ");
            int[] newLayerSize = new int[newLayerSizeString.length];
            double[][][] newWeight = new double[newLayerSizeString.length][][];     //[layer][neuron][prevNeuron]
            double[][] newBias = new double[newLayerSizeString.length][];     //[layer][neuron]

            //parse String[] to int[]
            for (int i = 0; i < newLayerSizeString.length; i++) {
                newLayerSize[i] = Integer.parseInt(newLayerSizeString[i]);
            }

            //Read weights
            for (int layer = 1; layer < newLayerSize.length; layer++) {
                newWeight[layer] = new double[newLayerSize[layer]][];
                newBias[layer] = new double[newLayerSize[layer]];
                for (int i = 0; i < newLayerSize[layer]; i++) {
                    if (scanner.next().equals("neuron["+layer+"]["+i+"]:")) {
                        String[] newWeightString = scanner.nextLine().split(",");
                        newWeight[layer][i] = new double[newWeightString.length];

                        //parse String[] to double[]
                        for (int j = 0; j < newWeightString.length; j++) {
                            newWeight[layer][i][j] = Double.parseDouble(newWeightString[j]);
                        }

                        //Read bias
                        if (scanner.next().equals("bias:")) {
                            newBias[layer][i] = scanner.nextDouble();
                            scanner.nextLine();
                        }
                        else throw new IOException();
                    }
                    else throw new IOException();
                }
            }

            //initialize the network with the file data
            System.out.println("##Network initializing...");
            this.networkSize = newLayerSize.length;
            this.layerSize = newLayerSize;
            this.input = new double[newLayerSize[0]];   //initialize all inputs as 0
            this.target = null;                         //temporarily no targets
            this.bias = newBias;


            iniNeurons(newWeight);


        } catch(IOException e) {
            System.out.println("Fail to import " + filename);
            e.printStackTrace();
        }
    }

    private void iniNeurons(double[][][] weight) {
        neuron = new Neuron[networkSize][];
        //input layer initialization
        neuron[0] = new Neuron[layerSize[0]];
        for (int i = 0; i < layerSize[0]; i++) {
            neuron[0][i] = new InputNeuron(i, input[i]);
            System.out.print(neuron[0][i]);
        }
        System.out.println();

        //other layers initialization
        for (int layer = 1; layer < networkSize; layer++) {
            neuron[layer] = new Neuron[layerSize[layer]];
            for (int i = 0; i < layerSize[layer]; i++) {
                neuron[layer][i] = new Neuron(i, prevLayerOutput(layer), bias[layer][i], layer, layerSize);
                //initialize weights
                if (weight != null) {
                    neuron[layer][i].setInputWeight(weight[layer][i]);
                }
                System.out.print(neuron[layer][i]);
            }
            System.out.println();
        }
    }


    public void predict(double[] newInput) {
        this.input = newInput;
        for (int i = 0; i < layerSize[0]; i++) {
            if (neuron[0][i] instanceof InputNeuron) {
                ((InputNeuron)neuron[0][i]).setInput(newInput[i]);
            }
        }
        feedForward();
        printNetwork(-1);
    }

    public void train(int iteration) {
        System.out.println("##Training...");
        for (int i = 0; i < iteration; i++) {
            feedForward();
            backPropagation();
            printNetwork(i);
            //output as file:
            new NetworkFileOutput().printFile(layerSize, neuron);
        }

    }

    private void feedForward() {
        for (int layer = 1; layer < networkSize; layer++) {
            for (int i = 0; i < layerSize[layer]; i++) {
                neuron[layer][i].setInput(prevLayerOutput(layer));  //pass inputs
                neuron[layer][i].calculate();                       //update sum and output
            }
        }
    }

    private void backPropagation() {
        for (int layer = networkSize - 1; layer > 0; layer--) {
            for (int i = 0; i < layerSize[layer]; i++) {           //current neuron

                for (int j = 0; j < layerSize[layer-1]; j++) {     //previous neuron
                    double deltaWeight = -learningRate * neuron[layer-1][j].getOutput() * delta(layer, i);
                    neuron[layer][i].updateInputWeight(j, deltaWeight);
                }

                neuron[layer][i].updateBias(-learningRate * delta(layer, i));
            }
        }
    }

    private double delta(int layer, int neuIndex) {
        //output layer
        if (layer == networkSize - 1) {
            double error = (neuron[layer][neuIndex].getOutput() - target[neuIndex]);
            return error * neuron[layer][neuIndex].getDerivOutput();
        }
        //inner layers
        else {
            double sum = 0;
            for (int i = 0; i < layerSize[layer + 1]; i++) {       // i: nextNeuron
                sum += (neuron[layer+1][i].getInputWeight()[neuIndex] * delta(layer + 1, i) );
            }
            return sum * neuron[layer][neuIndex].getDerivOutput();
        }
    }


    private double[] prevLayerOutput(int currentLayer) {
        int prevLayer = currentLayer - 1;
        double[] prevOutput = new double[layerSize[prevLayer]];
        for (int i = 0; i < layerSize[prevLayer]; i++) {
            prevOutput[i] = neuron[prevLayer][i].getOutput();
        }

        return prevOutput;
    }

    public void printNetwork(int iteration) {
        System.out.println("##Network [" + iteration + "]:");
        //print input layer first
        for (int i = 0; i < layerSize[0]; i++)
            System.out.print(neuron[0][i]);
        System.out.println();
        //print later layers
        for (int layer = 1; layer < networkSize; layer++) {
            for (int i = 0; i < layerSize[layer]; i++) {
                System.out.print(neuron[layer][i]);
            }
            System.out.println();
        }
    }

    public void acceptSample(double[] newInput, double[] newTarget) throws Exception{
        if (newInput.length != layerSize[0] || target.length != layerSize[networkSize-1]) {
            System.out.println("Invalid size of inputs/targets; input : [" + layerSize[0] + "] | target : [" + layerSize[networkSize-1] +"]");
            throw new Exception();
        }
        else {
            //update inputs
            this.input = newInput;
            for (int i = 0; i < layerSize[0]; i++) {
                if (neuron[0][i] instanceof InputNeuron) {
                    ((InputNeuron)neuron[0][i]).setInput(newInput[i]);
                }
            }
            //update targets
            this.target = newTarget;

        }
    }

    public void importSample(String filename) {
        try {
            Scanner scanner = new Scanner(new File(filename));



            //check if the file is blank
            if (!scanner.hasNext()) throw new IOException();

            //first line is sample size
            int sampleSize = scanner.nextInt();
            scanner.nextLine(); //skip a line
            sampleInput = new double[sampleSize][];        //[sample][input]
            sampleTarget = new double[sampleSize][];       //[sample][target]

            while(scanner.hasNextLine()){
                String[] aLine = scanner.nextLine().split("\\[|]: |, ");


                //check if the format right
                if (aLine.length >= 2) {
                    int index = Integer.parseInt(aLine[1]);


                    if (aLine[0].equals("input")) {
                        sampleInput[index] = new double[aLine.length - 2];
                        for (int i = 0; i < aLine.length - 2; i++) {
                            sampleInput[index][i] = Double.parseDouble(aLine[i + 2]);
                        }
                    }
                    else if (aLine[0].equals("target")) {
                        sampleTarget[index] = new double[aLine.length - 2];
                        for (int i = 0; i < aLine.length - 2; i++) {
                            sampleTarget[index][i] = Double.parseDouble(aLine[i + 2]);
                        }
                    }
                    else throw new IOException();

                }
                else throw new IOException();
            }





        } catch(IOException e) {
            System.out.println("Fail to import sample " + filename);
            e.printStackTrace();
        }

    }

    public void trainSample(int iteration) {
        for (int iter = 0; iter < iteration; iter++) {
            for (int sample = 0; sample < sampleInput.length; sample++) {
                try {
                    acceptSample(sampleInput[sample], sampleTarget[sample]);
                    train(1);
                } catch(Exception e) {
                    System.out.println("Fail to accept sample[" + sample + "]");
                    e.printStackTrace();
                }
            }
        }
    }

    public void setTarget(double[] target) {
        this.target = target;
    }

    public double[] getOutput() {
        int lastLayerIndex = networkSize - 1;
        double[] output = new double[layerSize[lastLayerIndex]];
        for (int i = 0; i < layerSize[lastLayerIndex]; i++) {
            output[i] = neuron[lastLayerIndex][i].getOutput();
        }
        return output;
    }

    public static double sigmoid(double x ,boolean deriv) {
        if (!deriv)
            return 1/(1+Math.exp(-x));
        else return sigmoid(x, false) * (1 - sigmoid(x, false));
    }

}
