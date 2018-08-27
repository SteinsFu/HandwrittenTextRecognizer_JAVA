import java.io.IOException;
import java.io.PrintStream;

public class NetworkFileOutput {

    private PrintStream p;

    public NetworkFileOutput() {
        try {
            p = new PrintStream("network.txt");
        } catch (IOException e) {
            System.out.println(e);
        }
    }

    public NetworkFileOutput(PrintStream p) {
        this.p = p;
    }

    public void printFile(int[] layerSize, Neuron[][] neuron) {
        //Network layerSize
        for (int item : layerSize) {
            p.print(item + " ");
        }
        p.println();

        //Network weights
        for (int layer = 1; layer < layerSize.length; layer++) {
            for (int i = 0; i < layerSize[layer]; i++) {
                p.print("neuron["+layer+"]["+i+"]: ");
                for (int weightIndex = 0; weightIndex < neuron[layer][i].getInputWeight().length; weightIndex++) {
                    //weight
                    p.print(neuron[layer][i].getInputWeight()[weightIndex]);
                    //comma ", "
                    if (weightIndex < neuron[layer][i].getInputWeight().length - 1) {
                        p.print(", ");
                    }
                }
                p.print("\nbias: " + neuron[layer][i].getBias());
                p.println();
            }
            p.println();
        }

    }


}
