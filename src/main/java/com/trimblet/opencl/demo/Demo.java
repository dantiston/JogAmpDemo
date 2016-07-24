package com.trimblet.opencl.demo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.Random;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;
import com.trimblet.opencl.constants.OpenClConstants;
import com.trimblet.opencl.obj.OpenCLContext;

public class Demo {

	private static final Logger LOG = LogManager.getLogger();

	public static final void main(String[] args) {

		// set up (uses default CLPlatform and creates context for all devices)
		// always make sure to release the context under all circumstances
		// not needed for this particular sample but recommented
		try (OpenCLContext context = OpenCLContext.create()) {
			LOG.info("Created context: {}", context);

			// select fastest device
			CLDevice device = context.getMaxFlopsDevice();
			LOG.info("Using device: {}", device);

			// create command queue on device.
			CLCommandQueue queue = device.createCommandQueue();

			int elementCount = 1_444_477; // Length of arrays to process
			int localWorkSize = Math.min(device.getMaxWorkGroupSize(), 256); // Local work size dimensions
			int globalWorkSize = roundUp(localWorkSize, elementCount); // rounded up to the nearest multiple of the localWorkSize

			// load sources, create and build program
			CLProgram program;
			try (InputStream source = Demo.class.getResourceAsStream(OpenClConstants.PROGRAM_FILE_NAME)) {
				program = context.createProgram(source).build();
			} catch (IOException e) {
				LOG.error(String.format("Couldn't find specified kernel: %s; ", OpenClConstants.PROGRAM_FILE_NAME), e);
				e.printStackTrace();
				program = null;
				System.exit(1);
			}

			// A, B are input buffers, C is for the result
			CLBuffer<FloatBuffer> inputVector1 = context.createFloatBuffer(globalWorkSize, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> inputVector2 = context.createFloatBuffer(globalWorkSize, Mem.READ_ONLY);
			CLBuffer<FloatBuffer> outputVector = context.createFloatBuffer(globalWorkSize, Mem.WRITE_ONLY);

			LOG.info("used device memory: {}MB",
					(inputVector1.getCLSize() + inputVector2.getCLSize() + outputVector.getCLSize()) / 1_000_000);

			// fill input buffers with random numbers
			// (just to have test data; seed is fixed -> results will not change between runs).
			fillBuffer(inputVector1.getBuffer(), 12_345);
			fillBuffer(inputVector2.getBuffer(), 67_890);

			// get a reference to the kernel function with the specified name
			// and map the buffers to its input parameters.
			CLKernel kernel = program.createCLKernel(OpenClConstants.PROGRAM_NAME);
			kernel.putArgs(inputVector1, inputVector2, outputVector).putArg(elementCount);

			// asynchronous write of data to GPU device,
			// followed by blocking read to get the computed results back.
			long time = System.nanoTime();
			queue.putWriteBuffer(inputVector1, false)
					.putWriteBuffer(inputVector2, false)
					.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize)
					.putReadBuffer(outputVector, true);
			time = System.nanoTime() - time;

			LOG.info("Finished computation in {}ms", time / 1_000_000);

			// print first few elements of the resulting buffer to the console.
			System.out.print("a + b = c results snapshot:");
			for (int i = 0; i < 10; i++) {
				System.out.print(outputVector.getBuffer().get() + ", ");
			}
			System.out.print("...; " + outputVector.getBuffer().remaining() + " more");

		} catch (Exception e) {
			LOG.error("Caught error while processing kernel: ", e);
			e.printStackTrace();
		}

	}

	private static void fillBuffer(FloatBuffer buffer, int seed) {
		Random rnd = new Random(seed);
		while (buffer.remaining() != 0) {
			buffer.put(rnd.nextFloat() * 100);
		}
		buffer.rewind();
	}

	private static int roundUp(int groupSize, int globalSize) {
		int r = globalSize % groupSize;
		if (r == 0) {
			return globalSize;
		} else {
			return globalSize + groupSize - r;
		}
	}

}
