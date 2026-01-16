"""
測試 C++ 代碼編譯和執行功能
"""
import pytest
from core import compile_and_execute_cpp, normalize_output


class TestCompileAndExecuteCpp:
    """測試 C++ 編譯和執行"""

    def test_simple_hello_world(self):
        """測試簡單的 Hello World 程序"""
        code = """
#include <iostream>
int main() {
    std::cout << "Hello World" << std::endl;
    return 0;
}
"""
        result = compile_and_execute_cpp(code, "", timeout=2)

        assert result['status'] == 'AC'
        assert normalize_output(result['output']) == 'Hello World'

    def test_simple_input_output(self):
        """測試簡單的輸入輸出"""
        code = """
#include <iostream>
int main() {
    int a, b;
    std::cin >> a >> b;
    std::cout << a + b << std::endl;
    return 0;
}
"""
        test_input = "3 5"
        result = compile_and_execute_cpp(code, test_input, timeout=2)

        assert result['status'] == 'AC'
        assert normalize_output(result['output']) == '8'

    def test_compilation_error(self):
        """測試編譯錯誤"""
        code = """
#include <iostream>
int main() {
    std::cout << "Missing semicolon"  // 缺少分號
    return 0;
}
"""
        result = compile_and_execute_cpp(code, "", timeout=2)

        assert result['status'] == 'CE'
        assert 'error' in result

    def test_runtime_error(self):
        """測試運行時錯誤"""
        code = """
#include <iostream>
int main() {
    int arr[1];
    arr[100] = 5;  // 數組越界
    return 0;
}
"""
        result = compile_and_execute_cpp(code, "", timeout=2)

        # 可能是 RTE 或 AC（取決於系統）
        assert result['status'] in ['RTE', 'AC']

    def test_timeout(self):
        """測試超時"""
        code = """
#include <iostream>
int main() {
    while(true) {}  // 無限循環
    return 0;
}
"""
        result = compile_and_execute_cpp(code, "", timeout=1)

        assert result['status'] == 'TLE'

    def test_wrong_answer(self):
        """測試錯誤答案（輸出不匹配）"""
        code = """
#include <iostream>
int main() {
    std::cout << "42" << std::endl;
    return 0;
}
"""
        result = compile_and_execute_cpp(code, "", timeout=2)

        # 這個測試只驗證代碼能執行，WA 判定在更高層
        assert result['status'] == 'AC'
        assert normalize_output(result['output']) == '42'

    def test_bits_stdc_header(self):
        """測試 bits/stdc++.h 頭文件（Competitive Programming 常用）"""
        code = """#include <bits/stdc++.h>
using namespace std;

int main() {
    int t;
    cin >> t;
    while (t--) {
        int x, y;
        cin >> x >> y;
        int a = min(x, y);
        int b = max(x, y);
        cout << a << ' ' << b << '\\n';
    }
    return 0;
}"""
        test_input = "3\n1 9\n8 4\n2 0"
        expected_output = "1 9\n4 8\n0 2"

        result = compile_and_execute_cpp(code, test_input, timeout=2)

        assert result['status'] == 'AC', f"Compilation failed: {result.get('compile_error', result.get('error'))}"
        assert normalize_output(result['output']) == normalize_output(expected_output)


class TestNormalizeOutput:
    """測試輸出標準化"""

    def test_strip_trailing_whitespace(self):
        """測試去除尾部空白"""
        assert normalize_output("hello   ") == "hello"
        assert normalize_output("   hello") == "hello"

    def test_strip_trailing_newlines(self):
        """測試去除尾部換行"""
        assert normalize_output("hello\n\n") == "hello"
        assert normalize_output("hello\n") == "hello"

    def test_normalize_line_endings(self):
        """測試標準化行尾"""
        assert normalize_output("line1  \nline2  ") == "line1\nline2"
        assert normalize_output("  line1\n  line2  ") == "line1\nline2"

    def test_multiple_lines(self):
        """測試多行輸出"""
        output = "1 9\n4 8\n1 4"
        expected = "1 9\n4 8\n1 4"
        assert normalize_output(output) == expected
