import java.util.Scanner;

public class RecognizerDemo {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        //image size : 25x28
        Recognizer recognizer = new Recognizer(20);

        System.out.println("Train? Y/y for yes");
        if (scanner.nextLine().toLowerCase().equals("y"))
            recognizer.train(20);

        //user input loop:
        while(true) {
            System.out.print("predict (image path): ");
            String imgPath = scanner.nextLine();
            recognizer.predict(imgPath);
        }

    }

}
