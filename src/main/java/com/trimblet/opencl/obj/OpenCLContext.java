package com.trimblet.opencl.obj;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLMemory.Mem;
import com.jogamp.opencl.CLProgram;

public final class OpenCLContext implements AutoCloseable {

	private final CLContext context;

	public OpenCLContext(CLContext context) {
		if (context == null) {
			throw new NullPointerException();
		}
		this.context = context;
	}

	public static final OpenCLContext create() {
		return new OpenCLContext(CLContext.create());
	}

	public final CLDevice getMaxFlopsDevice() {
		return this.context.getMaxFlopsDevice();
	}

	public final CLProgram createProgram(InputStream stream) throws IOException {
		return this.context.createProgram(stream);
	}

	public final CLBuffer<FloatBuffer> createFloatBuffer(int size, Mem... location) {
		return this.context.createFloatBuffer(size, location);
	}

	@Override
	public final void close() throws Exception {
		this.context.release();
	}

	@Override
	public final String toString() {
		return this.context.toString();
	}

}
