public class InputNeuron extends Neuron {


    public InputNeuron(int index, double input) {
        super(index);
        output = input;
    }


    public void setInput(double input) {
        this.output = input;
    }

}
