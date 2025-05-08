import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { invoice_id } = body;

    if (!invoice_id) {
      return NextResponse.json({ error: 'invoice_id is required' }, { status: 400 });
    }

    // Get the workspace root. process.cwd() should give the project root in a Next.js environment.
    const workspaceRoot = process.cwd();
    const pythonScriptPath = path.join(workspaceRoot, 'scripts', 'run_find_duplicates.py');
    const pythonExecutable = 'python3'; // Or just 'python' depending on the system setup

    return new Promise((resolve, reject) => {
      // Set PYTHONPATH to the workspace root so that Python can find modules in 'src'
      const options = { env: { ...process.env, PYTHONPATH: workspaceRoot } };
      const pythonProcess = spawn(pythonExecutable, [pythonScriptPath, invoice_id], options);

      let stdoutData = '';
      let stderrData = '';

      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.error(`Python Script STDERR: ${data.toString()}`); // Log Python errors to Next.js console
      });

      pythonProcess.on('close', (code) => {
        console.log(`Python script exited with code ${code}`);
        if (stderrData && code !== 0) {
          // Prefer stderr if it has content and exit code is non-zero
          try {
            // Attempt to parse stderr in case it's our JSON error message
            const errorOutput = JSON.parse(stderrData.trim().split('\n').pop() || stderrData.trim()); 
            resolve(NextResponse.json(errorOutput, { status: 500 }));
          } catch (e) {
            resolve(NextResponse.json({ error: 'Python script execution failed.', details: stderrData }, { status: 500 }));
          }
          return;
        }
        
        try {
          // Try to parse the last line of stdout as JSON
          // This handles cases where stderr might have progress messages but stdout has the final JSON output
          const lines = stdoutData.trim().split('\n');
          const lastLine = lines[lines.length -1];
          const result = JSON.parse(lastLine);

          if (result.error && code !== 0) {
             resolve(NextResponse.json(result, { status: 500 }));
          } else if (result.error) { // Error reported in JSON but script exited 0 (e.g. ID not found)
             resolve(NextResponse.json(result, { status: 404 })); // Or appropriate status
          }else {
            resolve(NextResponse.json(result, { status: 200 }));
          }
        } catch (error) {
          console.error('Error parsing Python script output:', error);
          console.error('Python stdoutData:', stdoutData);
          console.error('Python stderrData:', stderrData);
          resolve(NextResponse.json({ error: 'Failed to parse output from duplicate detection script.', raw_stdout: stdoutData, raw_stderr: stderrData }, { status: 500 }));
        }
      });

      pythonProcess.on('error', (err) => {
        console.error('Failed to start Python script:', err);
        resolve(NextResponse.json({ error: 'Failed to start duplicate detection script.', details: err.message }, { status: 500 }));
      });
    });

  } catch (error: any) {
    console.error('API Error:', error);
    return NextResponse.json({ error: 'Internal server error.', details: error.message }, { status: 500 });
  }
} 