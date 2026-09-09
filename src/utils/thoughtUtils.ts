/**
 * Utility functions for cleaning and formatting AI thought streams.
 * Removes action separators, raw JSON blocks, XML tags, stray formatting artifacts,
 * and collapses repetitive content.
 */

export function cleanThoughtText(rawText: string): string {
  if (!rawText) return '';

  let text = rawText;

  // 1. Cut off action separator and everything after it
  const actionSepIndex = text.indexOf('---ACTION---');
  if (actionSepIndex !== -1) {
    text = text.substring(0, actionSepIndex);
  }

  // 2. Remove code blocks / JSON blocks
  text = text.replace(/```(?:json|action)?[\s\S]*?```/gi, '');
  text = text.replace(/```[\s\S]*$/gi, ''); // Trailing unclosed codeblock

  // 3. Remove standalone JSON objects if they appear at the end
  text = text.replace(/\{[\s\n\r]*"(?:memory_write|actions|gaze_target|new_motor_program|program_sequence)"[\s\S]*$/i, '');

  // 4. Remove XML tags like <action>, </action>, <speak>, </speak>, <thought>, </thought>
  text = text.replace(/<\/?(?:action|thought|speak|json|response|output)\b[^>]*>/gi, '');

  // 5. Remove leading/trailing prefixes like "Thought:", "Thinking:", "Cognition:"
  text = text.replace(/^(?:(?:\[?LIVE\]?|\*?\*?(?:Thought|Thinking|Cognition|Internal Monologue)\*?\*?:?)\s*)+/i, '');

  // 6. Clean up raw markdown headers like "### Action JSON"
  text = text.replace(/#{1,6}\s*(?:Action|Actions|JSON|Motor Commands)[\s\S]*$/i, '');

  // 7. Strip leftover action label tags
  text = text.replace(/(?:Action JSON|Motor Plan|Output JSON):?\s*$/i, '');

  // 8. Clean up multiple excessive backticks
  text = text.replace(/`{1,3}/g, '');

  // 9. Collapse repetitive sentences (same sentence repeated 2+ times)
  text = collapseRepetitions(text);

  // 10. Normalize multiple blank lines into clean paragraph breaks
  text = text.replace(/\r\n/g, '\n').replace(/\n{3,}/g, '\n\n');

  return text.trim();
}

/**
 * Detects and collapses repeated sentences or near-duplicate phrases.
 * If the same sentence (normalized) appears 2+ times consecutively, keep only the first occurrence.
 */
function collapseRepetitions(text: string): string {
  const sentences = text.split(/(?<=[.!?])\s+/);
  if (sentences.length < 2) return text;

  const normalized = (s: string) =>
    s.toLowerCase().replace(/[^\w\s]/g, '').replace(/\s+/g, ' ').trim();

  const result: string[] = [];
  let lastNorm = '';
  let repeatCount = 0;

  for (const sentence of sentences) {
    const norm = normalized(sentence);
    if (norm === lastNorm && norm.length > 10) {
      repeatCount++;
      if (repeatCount === 1) {
        // Second occurrence — mark as repetition
        result.push('[repeated]');
      }
      // Skip further duplicates
      continue;
    }
    // Also catch near-duplicates (80%+ word overlap)
    if (lastNorm.length > 10 && norm.length > 10) {
      const lastWords = new Set(lastNorm.split(' '));
      const curWords = norm.split(' ');
      const overlap = curWords.filter(w => lastWords.has(w)).length;
      const overlapRatio = overlap / Math.max(curWords.length, 1);
      if (overlapRatio > 0.8 && curWords.length > 3) {
        repeatCount++;
        if (repeatCount === 1) {
          result.push('[repeated]');
        }
        continue;
      }
    }
    repeatCount = 0;
    lastNorm = norm;
    result.push(sentence);
  }

  return result.join(' ').replace(/\[repeated\]\s*\[repeated\]*/g, '[repeated]');
}
