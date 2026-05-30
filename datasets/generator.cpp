#include <bits/stdc++.h>
using namespace std;
#define PROMPT_SKIP cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); continue

struct Prompter {
    string join(const vector<string>& v, const string s){
        string restr = "";
        for(int _i = 0; _i < v.size(); _i++){
            restr += "[" + to_string(_i+1) + "] " + v[_i];
            if(_i < v.size()-1) restr += s;
        }
        return restr;
    }
    int prompt_for_int(const string& message) {
        int res;
        while(true){
            cout << message << ": ";
            if(cin >> res){
                return res;
            } else {
                PROMPT_SKIP;
            }
        }
    }
    string prompt_options(const string& message, const vector<string>& opts){
        int res;
        while(true){
            cout << message << " (" << join(opts," | ") << "): ";
            if(cin >> res){
                if(res > opts.size() || res <= 0) continue;
                return opts[res-1];
            } else {
                PROMPT_SKIP;
            }
        }
    }
};

struct MazeGenerator {

};

int main(){
    Prompter p;
}