package unused;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;

public class TestLocal {

    public static void main(String[] args) {
        try {
            ProcessBuilder pb = new ProcessBuilder("python", "C:\\Users\\Jonas\\IdeaProjects\\PaddleOCR\\test5.py");
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedWriter pyIn = new BufferedWriter(
                    new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            BufferedReader pyOut = new BufferedReader(
                    new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));

            String line;
            System.out.println("1");
            pyIn.write("C:\\Users\\Jonas\\Downloads\\EnT2\\tmp");
            pyIn.write("|");
            pyIn.write("3");
            pyIn.newLine();
            while (!pyOut.readLine().trim().equals("done"));
            System.out.println("1s");
            Thread.sleep(30000);
            System.out.println("2");
            pyIn.write("C:\\Users\\Jonas\\Downloads\\EnT2\\tmp");
            pyIn.write("|");
            pyIn.write("5");
            pyIn.newLine();
            while (!pyOut.readLine().trim().equals("done"));
            System.out.println("2s");
            Thread.sleep(30000);
            System.out.println("3");
            pyIn.write("C:\\Users\\Jonas\\Downloads\\EnT2\\tmp");
            pyIn.write("|");
            pyIn.write("7");
            pyIn.newLine();
            while (!pyOut.readLine().trim().equals("done"));
            System.out.println("4");
            int exitCode = process.waitFor();
        } catch (Exception e) {
            e.printStackTrace(System.out);
        }

    }
}
