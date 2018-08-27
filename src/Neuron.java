import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

public class Neuron {
    protected int index;
    protected double output;

    private double[] input;          //[prevNeuron]  previous neurons outputs
    private double[] inputWeight;    //[prevNeuron]
    private double bias;
    private double sum;
    private int layer;
    private int[] LAYER_SIZE;

    Random r = new Random();

    //constructor for InputNeuron
    protected Neuron(int index) {
        this.index = index;
    }

    //constructor
    public Neuron(int index, double[] input, double bias, int layer, int[] LAYER_SIZE) {
        this.index = index;
        this.input  = input;
        this.inputWeight = inputWeight;
        this.bias = bias;
        this.layer = layer;
        this.LAYER_SIZE = LAYER_SIZE;

        initWeight();
        calculate();


    }

    public void calculate() {
        sum = bias;
        for (int prevNeuron = 0; prevNeuron < input.length; prevNeuron++) {
            sum += input[prevNeuron] * inputWeight[prevNeuron];
        }

        output = Network.sigmoid(sum, false);
    }

    private void initWeight() {

        inputWeight = new double[LAYER_SIZE[layer-1]];
        for (int i = 0; i < inputWeight.length; i++) {
            inputWeight[i] = r.nextDouble()/2 - 0.25;    //-0.25~0.25
        }
    }

    //for random bias
    private void initBias() {
        bias = r.nextDouble();
    }



    //getter and setter
    public double[] getInput() {
        return input;
    }

    public double[] getInputWeight() {
        return inputWeight;
    }

    public double getBias() {
        return bias;
    }

    public double getSum() {
        return sum;
    }

    public double getOutput() {
        return output;
    }

    public double getDerivOutput() {
        return Network.sigmoid(sum, true);
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public void setInputWeight(double[] inputWeight) {
        this.inputWeight = inputWeight;
    }

    public void updateInputWeight(int prevNeuron, double deltaWeight) {
        this.inputWeight[prevNeuron] += deltaWeight;
    }

    public void updateBias(double deltaBias) {
        this.bias += deltaBias;
    }


    public String toString() {
        DecimalFormat f = new DecimalFormat("0.0000");
        return "Neuron[" + layer + "][" + index + "] { b: " + bias + "| w: " + Arrays.toString(inputWeight) +
                " }\t\t(" + f.format(sum) + " | " + f.format(output) +")\n";
    }

}
