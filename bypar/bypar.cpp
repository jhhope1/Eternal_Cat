#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <iostream>
#include <string.h>
#include <vector>
#include <math.h>
#include <algorithm>
using namespace std;
typedef long long ll;
typedef pair<int, string> pis;
typedef unordered_map<string, int> SI_map;
unordered_map<int, SI_map> V[1000000];
vector<string> target[1000000];
vector<string> pred[1000000];
int to_int(string s){
    int n = s.size();
    int a = 1;
    int re = 0;
    for(int i=n-1 ; i>=0 ; i--){
        re+=a*(s[i]-'0');
    }
    return re;
}
vector<string> tag_input(string str){
    int n = str.size();
    string T;
    vector<string> re;
    for(int i=0 ; i<n ; i++){
        if(str[i] == ','){
            if(T.size()){
                re.push_back(T);
            }
            T.clear();
        }
        else if(str[i]!=' '){
            T.push_back(str[i]);
        }
    }
    if(T.size()){
        re.push_back(T);
        T.clear();
    }
    return re;
}
vector<int> list_input(string str){
    int n = str.size();
    string T;
    vector<int> re;
    for(int i=0 ; i<n ; i++){
        if(str[i]==','){
            if(T.size()){
                int a = to_int(T);
                re.push_back(a);
            }
            T.clear();
        }
        else if(str[i]!=' '){
            T.push_back(str[i]);
        }
    }
    if(T.size()){
        re.push_back(to_int(T));
        T.clear();
    }
    return re;
}
void add_Edge(vector<string> TI, vector<int> LI, int l){
    int n = LI.size();
    for(int i=0 ; i<n ; i++){
        for(int j=i+1 ; j<n ; j++){
            int a = LI[i];
            int b = LI[j];
            for(string s: TI){
                if(V[a].find(b) ==V[a].end()){
                    V[a].insert(make_pair(b,SI_map()));
                    V[b].insert(make_pair(a,SI_map()));
                    V[a][b].insert(make_pair(s, l));
                    V[b][a].insert(make_pair(s, l));
                }
                else{
                    if(V[a][b].find(s)==V[a][b].end()){
                        V[a][b].insert(make_pair(s,l));
                        V[b][a].insert(make_pair(s,l));
                    }
                    else{
                        V[a][b][s] = V[a][b][s]+l;
                        V[b][a][s] = V[b][a][s]+l;
                    }
                }
            }
        }
    }
}
void train_input(){//tag, list, like
    std::ifstream in("train.txt");
    if(!in.is_open()){
        cout<<"train.txt is not opened\n";
        return;
    }
    string str;
    int n = 50000;
    //cin>>n;
    int l;
    for(int i=0 ; i<n ; i++){
        getline(cin, str);
        vector<string> TI = tag_input(str);
        getline(cin, str);
        vector<int> LI = list_input(str);
        cin>>l;
        add_Edge(TI,LI,l);
    }
    in.close();
}
vector<string> query(){
    string str;
    getline(cin,str);
    vector<int> LI = list_input(str);
    int n = LI.size();
    SI_map re;
    for(int i=0 ; i<n ; i++){
        for(int j=i+1 ; j<n ; j++){
            int a = LI[i];
            int b = LI[j];
            if(V[a].find(b)!=V[a].end()){
                for(auto p: V[a][b]){
                    if(re.find(p.first)!=re.end()){
                        re.find(p.first)->second+=p.second;
                    }
                    else{
                        re.insert(p);
                    }
                }
            }
        }
    }
    vector<pis> ans;
    for(auto A:re){
        ans.push_back(pis(A.second,A.first));
    }
    sort(ans.begin(),ans.end(),greater<pis>());
    vector<string> S;
    for(int i=0 ; i<max(int(ans.size()),10) ; i++){
        S.push_back(ans[i].second);
    }
    return S;
}

void test(){
    std::ifstream in("test.txt");
    if(!in.is_open()){
        cout<<"test.txt is not opened\n";
        return;
    }
    int n = 100;
    //cin>>n;
    for(int i=0 ; i<n ; i++){
        vector<string> tag = query();
        for(string s:tag){
            pred[i].push_back(s);
        }
    }
    in.close();
}
void target_input(){
    std::ifstream in("target.txt");
    if(!in.is_open()){
        cout<<"target.txt is not opened\n";
        return;
    }
    int n = 100;
    //cin>>n;
    for(int i=0 ; i<n ; i++){
        string str;
        getline(cin,str);
        vector<string> TI = tag_input(str);
        for(string s:TI){
            target[i].push_back(s);
        }
    }
    in.close();
}
double NDCG(){
    int n = 100;
    //
    double re = 0;
    for(int i=0 ; i<n ; i++){
        double a = 0;
        int cnt = 1;
        for(string p:pred[i]){
            cnt++;
            for(string t:target[i]){
                if(t==p){
                    a+=1/log(cnt);
                    break;
                }
            }
        }
        re+=a/(cnt-1);
    }
    return re/n;
}
void eval(){
    target_input();
    std::ifstream in("NDCG.txt");
    if(!in.is_open()){
        cout<<"NDCG.txt is not opened\n";
        return;
    }
    cout<<NDCG();
    in.close();
}

int main(){
    //ios_base::sync_with_stdio(0);cin.tie(0);cout.tie(0);
    train_input();
    test();
    eval();
}