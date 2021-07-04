import java.util.*;
import java.io.*;

public class test {
    protected static String pmzReadLine(String value, float[] x_data) {
        String raw_x = value.split("\t", 2)[1];
        for(String s: raw_x.split(" ")) {
            int idx = Integer.parseInt(s.split(":")[0]);
            float val = Float.parseFloat(s.split(":")[1]);
            x_data[idx] = val;
        }
        return value.substring(0, 2);
    }

    public static void main(String[] args) {
        String y_data = "";
        float[] x_data = new float[100];
        String value = "体育1220.txt\t12:0.342 43:0.21 99:0.99";
        y_data = pmzReadLine(value, x_data);
        System.out.println(y_data);
        for(float f: x_data) System.out.printf("%f ", f);
    }
}